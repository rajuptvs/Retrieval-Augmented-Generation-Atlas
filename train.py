# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import uuid
import os
import time
from collections import defaultdict
import wandb
import numpy as np
import torch
import torch.cuda
import logging
from evaluate import evaluate
from src import dist_utils, slurm, util
from src.index_io import load_or_initialize_index, save_embeddings_and_index
from src.model_io import create_checkpoint_directories, load_or_initialize_atlas_model, save_atlas_model
from src.options import get_options
from src.tasks import get_task
from wandb import plot

os.environ["TOKENIZERS_PARALLELISM"] = "true"
GRAD_SCALE_UPPER_BOUND_MEAN: int = 1000
GRAD_SCALE_LOWER_BOUND_MEAN: float = 0.01
THRESHOLD_GRAD_STATS: int = 100

logger = logging.getLogger(__name__)

print(torch.cuda.is_available())
def train(
    model,
    index,
    passages,
    optimizer,
    scheduler,
    retr_optimizer,
    retr_scheduler,
    step,
    opt,
    checkpoint_path,
):
    data=[]
    id=str(uuid.uuid4())
    wandb.init(project='Split Test', name=opt.name)
    tb_logger = util.init_tb_logger(os.path.join(opt.checkpoint_dir, opt.name), is_main=opt.is_main)
    run_stats = util.WeightedAvgStats()
    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)

    # different seed for different sampling depending on global_rank
    torch.manual_seed(opt.global_rank + opt.seed)
    wandb.watch(model, log="all")
    scale = 2.0
    grad_stats = defaultdict(lambda: [])
    task = get_task(opt, unwrapped_model.reader_tokenizer)
    index_refresh_scheduler = util.IndexRefreshScheduler(
        opt.refresh_index, opt.freeze_retriever_steps, opt.train_retriever
    )
    while step < opt.total_steps:
        data_iterator = task.data_iterator(
            opt.train_data, opt.global_rank, opt.world_size, repeat_if_less_than_world_size=True, opt=opt
        )
        data_iterator = filter(None, map(task.process, data_iterator))
        data_iterator = task.batch_iterator(data_iterator, opt.per_gpu_batch_size, drop_last=True, shuffle=opt.shuffle)
        for i, batch in enumerate(data_iterator):

            iter_stats = {}
            model.train()
            if not opt.use_file_passages and index_refresh_scheduler.is_time_to_refresh(step):

                if not (step == 0 and opt.load_index_path is not None):  # Dont refresh index if just loaded it
                    indexing_start = time.time()
                    unwrapped_model.build_index(index, passages, opt.per_gpu_embedder_batch_size, logger)
                    iter_stats["runtime/indexing"] = (time.time() - indexing_start, 1)

                    if opt.save_index_path is not None:
                        save_embeddings_and_index(index, opt)
            step += 1
            train_step_start = time.time()

            reader_loss, retriever_loss = model(
                index=index,
                query=batch["query"],
                target=batch["target"],
                target_tokens=batch.get("target_tokens"),
                passages=batch["passages"] if opt.use_file_passages else None,
                batch_metadata=batch.get("metadata"),
                filtering_fun=task.filter,
                train_retriever=opt.train_retriever and step > opt.freeze_retriever_steps,
                iter_stats=iter_stats,
            )

            if retriever_loss is not None and opt.train_retriever:
                train_loss = reader_loss.float() + retriever_loss
            else:
                train_loss = reader_loss

            iter_stats["loss/train_loss"] = (train_loss.item(), len(batch["query"]))

            backward_start = time.time()
            train_loss = scale * train_loss
            train_loss.backward()
            iter_stats["runtime/backward"] = (time.time() - backward_start, 1)

            model_update_start = time.time()
            stats = util.compute_grad_stats(model)
            if stats["skip_example"]:
                model.zero_grad()
                # continue
            else:
                for k, v in stats.items():
                    grad_stats[k].append(v)

            if len(grad_stats["max"]) >= THRESHOLD_GRAD_STATS:
                if np.mean(grad_stats["max"]) > GRAD_SCALE_UPPER_BOUND_MEAN:
                    scale /= 2
                elif np.mean(grad_stats["mean"]) < GRAD_SCALE_LOWER_BOUND_MEAN:
                    scale *= 2
                # print(f'Scale: {scale}')
                grad_stats.clear()

            if step % opt.accumulation_steps == 0 and not stats["skip_example"]:
                if opt.is_distributed and opt.shard_optim:
                    optimizer.clip_grad_norm(scale * opt.clip)
                    if opt.train_retriever:
                        retr_optimizer.clip_grad_norm(scale * opt.clip)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), scale * opt.clip)

                optimizer.step(scale=scale)
                scheduler.step()
                if opt.train_retriever:
                    retr_optimizer.step(scale=scale)
                    retr_scheduler.step()
                model.zero_grad()
            iter_stats["runtime/model_update"] = (time.time() - model_update_start, 1)
            iter_stats["runtime/train_step"] = (time.time() - train_step_start, 1)
            run_stats.update(iter_stats)

            if step % opt.log_freq == 0:
                log = f"{step} / {opt.total_steps}"
                for k, v in sorted(run_stats.average_stats.items()):
                    log += f" | {k}: {v:.3g}"
                    if tb_logger:
                        tb_logger.add_scalar(k, v, step)
                log += f" | lr: {scheduler.get_last_lr()[0]:0.2g}"
                log += f" | Memory: {torch.cuda.max_memory_allocated()//1e9} GiB"
                if tb_logger:
                    tb_logger.add_scalar("lr", scheduler.get_last_lr()[0], step)

                logger.info(log)
                # print("------------------------------------------------- CHECK THE LOGS NOW ---------------------------------------------------------------")
                # logger.info(run_stats.average_stats["loss/reader_loss"])
                # metrics = evaluate(model, index, opt, data_path, step)
                wandb.log({
                "Step": step,
                "Loss": run_stats.average_stats["loss/train_loss"],
                "Reader Loss": run_stats.average_stats["loss/reader_loss"],
                "Retriever Loss": run_stats.average_stats["loss/retriever_loss"],
                "Learning Rate": scheduler.get_last_lr()[0],
                "Memory Utilizing": torch.cuda.max_memory_allocated()//1e9,
                })
                run_stats.reset()
                
                # metrics = evaluate(model, index, opt, data_path, step)
                
            if step % opt.eval_freq == 0:
                for data_path in opt.eval_data:
                    print("Data path: ", data_path)
                    dataset_name = os.path.basename(data_path)

                    metrics = evaluate(model, index, opt, data_path, step)
                    log_message = f"Dataset: {dataset_name}"
                    print("1 ", metrics)
                    for k, v in metrics.items():
                        log_message += f" | {v:.3f} {k}"
                        if tb_logger:
                            tb_logger.add_scalar(f"{dataset_name}/{k}", v, step)
                    wandb.log({
                            "Eval Loss": metrics["eval_loss"],
                            "F1": metrics["f1"],
                            "EM": metrics["exact_match"]
                            
                    
                    })
                    data.append([id,step,metrics["eval_loss"],metrics["f1"],metrics["exact_match"]])
                    print("saving the plots")
                    table = wandb.Table(data=data, columns=["id","step","eval_loss", "f1","exact_match"])
                    line_plot_eval_loss = plot.line(table, x='step', y='eval_loss', title='Line Plot_eval_loss')
                    line_plot_f1 = plot.line(table, x='step', y='f1', title='Line Plot_f1')
                    line_plot_exact_match = plot.line(table, x='step', y='exact_match', title='Line Plot_exact_match')
                    histogram_eval_loss = plot.histogram(table, value='eval_loss', title='Histogram_eval_loss')
                    histogram_f1 = plot.histogram(table, value='f1', title='Histogram_f1')
                    histogram_exact_match = plot.histogram(table, value='exact_match', title='Histogram_exact_match')
                    scatter_eval_loss = plot.scatter(table, x='step', y='eval_loss', title='Scatter Plot_eval_loss')
                    scatter_f1 = plot.scatter(table, x='step', y='f1', title='Scatter Plot_f1')
                    scatter_exact_match = plot.scatter(table, x='step', y='exact_match', title='Scatter Plot_exact_match')
                    wandb.log({'line_plot_eval_loss': line_plot_eval_loss, 
                            'line_plot_f1': line_plot_f1,
                                'line_plot_exact_match': line_plot_exact_match,
                                'histogram_eval_loss': histogram_eval_loss,
                                'histogram_f1': histogram_f1,
                                'histogram_exact_match': histogram_exact_match,
                                'scatter_eval_loss': scatter_eval_loss,
                                'scatter_f1': scatter_f1,
                                'scatter_exact_match': scatter_exact_match})
                    logger.info(log_message)

            if step % opt.save_freq == 0 and opt.is_main:
                save_atlas_model(
                    unwrapped_model,
                    optimizer,
                    scheduler,
                    retr_optimizer,
                    retr_scheduler,
                    step,
                    opt,
                    checkpoint_path,
                    f"step-{step}",
                )
            if step == opt.total_steps:
                pass
                # print("saving the plots")
                # table = wandb.Table(data=data, columns=["step","eval_loss", "f1","exact_match"])
                # line_plot_eval_loss = plot.line(table, x='step', y='eval_loss', title='Line Plot_eval_loss')
                # line_plot_f1 = plot.line(table, x='step', y='f1', title='Line Plot_f1')
                # line_plot_exact_match = plot.line(table, x='step', y='exact_match', title='Line Plot_exact_match')
                # histogram_eval_loss = plot.histogram(table, value='eval_loss', title='Histogram_eval_loss')
                # histogram_f1 = plot.histogram(table, value='f1', title='Histogram_f1')
                # histogram_exact_match = plot.histogram(table, value='exact_match', title='Histogram_exact_match')
                # scatter_eval_loss = plot.scatter(table, x='step', y='eval_loss', title='Scatter Plot_eval_loss')
                # scatter_f1 = plot.scatter(table, x='step', y='f1', title='Scatter Plot_f1')
                # scatter_exact_match = plot.scatter(table, x='step', y='exact_match', title='Scatter Plot_exact_match')
                # wandb.log({'line_plot_eval_loss': line_plot_eval_loss, 
                #            'line_plot_f1': line_plot_f1,
                #             'line_plot_exact_match': line_plot_exact_match,
                #             'histogram_eval_loss': histogram_eval_loss,
                #             'histogram_f1': histogram_f1,
                #             'histogram_exact_match': histogram_exact_match,
                #             'scatter_eval_loss': scatter_eval_loss,
                #             'scatter_f1': scatter_f1,
                #             'scatter_exact_match': scatter_exact_match})
                
            if step > opt.total_steps:
                exit()


if __name__ == "__main__":
    options = get_options()
    opt = options.parse()

    torch.manual_seed(opt.seed)
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()

    checkpoint_path, saved_index_path = create_checkpoint_directories(opt)

    logger = util.init_logger(opt.is_main, opt.is_distributed, os.path.join(checkpoint_path, "run.log"))
    if opt.is_main:
        options.print_options(opt)

    logger.info(f"world size: {dist_utils.get_world_size()}")

    index, passages = load_or_initialize_index(opt)
    model, optimizer, scheduler, retr_optimizer, retr_scheduler, opt, step = load_or_initialize_atlas_model(opt)

    if opt.is_distributed:
        if opt.shard_grads:
            import fairscale.nn.data_parallel

            model.reader = fairscale.nn.data_parallel.ShardedDataParallel(
                model.reader, optimizer, auto_refresh_trainable=False
            )
            if opt.train_retriever:
                model.retriever = fairscale.nn.data_parallel.ShardedDataParallel(
                    model.retriever, retr_optimizer, auto_refresh_trainable=False
                )
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[opt.local_rank],
                output_device=opt.local_rank,
                find_unused_parameters=True,
            )
            model._set_static_graph()

    logger.info("Start training")
    dist_utils.barrier()

    train(
        model,
        index,
        passages,
        optimizer,
        scheduler,
        retr_optimizer,
        retr_scheduler,
        step,
        opt,
        checkpoint_path,
    )
