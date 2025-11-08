# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from accelerate import Accelerator
from omegaconf import open_dict
from common_utils import (
    setup_basics,
    attention_prior_annealing
)
from plot_utils import (
    plot_alignment_to_figure,
    plot_spectrogram_to_figure,
    plot_to_figure
)
from logging_utils import Averager
from model_utils import (
    get_config,
    get_model,
    get_vocoder_model,
    get_dataloaders,
    get_optimizer,
    get_lr_scheduler,
    get_loss_function
)
import hydra
import torch
import time
import os


def maybe_save_checkpoint(accelerator, lr_scheduler, args):
    if (
        args.current_train_step > args.optim.total_steps
        or args.current_train_step % args.checkpoint.every_steps == 0
    ):
        output_dir = f'checkpoint-{args.mode}-{args.current_train_step}'
        accelerator.save_state(output_dir=output_dir)
        if accelerator.is_main_process:
            torch.save(lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.bin"))


def maybe_eval_predict(model, teacher_model, dataloader, logger, args, loss_function, vocoder_model, vocoder_hparams, accelerator):
    if (
        args.current_train_step > args.optim.total_steps
        or args.current_train_step % args.eval.every_steps == 0
    ):
        model.eval()

        with torch.no_grad():
            eval(model, teacher_model, dataloader, logger, args, loss_function, accelerator)
            predict(model, teacher_model, dataloader, logger, args, vocoder_model, vocoder_hparams, accelerator)

        args.last_log = time.time()
        model.train()


def eval(model, teacher_model, dataloader, logger, args, loss_function, accelerator):
    args.last_log = time.time()
    averager = Averager()
    image_index = 0

    for batch_id, batch in enumerate(dataloader, start=1):
        if batch_id == args.eval.steps * args.optim.grad_acc:
            break

        model_outputs = forward(model, batch, args, teacher_model)
        loss, stats = loss_function(model_outputs, batch)
        gathered_stats = accelerator.gather_for_metrics(stats)
        averaged_stats_across_gpus = average_stats(gathered_stats)
        averager.update(averaged_stats_across_gpus)

        if args.mode == "pretrain":
            (mel_pred, dec_mask, dur_pred, log_dur_pred, pitch_pred, pitch_tgt, energy_pred, energy_tgt, attn_soft, attn_hard, dur_tgt, attn_logprob), _ = model_outputs
            _, _, mel_tgt, _, _, _, _, in_lens, out_lens = batch
        elif args.mode == "distill":
            student_outputs, teacher_outputs = model_outputs
            (mel_pred, stu_dec_mask, dur_pred, log_dur_pred, pitch_pred, energy_pred), _ = student_outputs
            (mel_tgt, tea_dec_mask, dur_tgt, log_dur_tgt, pitch_tgt, energy_tgt), _ = teacher_outputs
            mel_tgt = mel_tgt.permute(0, 2, 1)
            symbol_ids, num_symbols_of_each_word, speaker_id, in_lens = batch
            out_lens = tea_dec_mask.squeeze(2).int().sum(1)
            attn_soft = None
            attn_hard = None
        else:
            raise NotImplementedError
        
        batch_size = mel_pred.shape[0]
        
        if image_index < args.eval.max_num_images:
            mel_pred = mel_pred.permute(0, 2, 1).detach().cpu().numpy()
            mel_tgt = mel_tgt.detach().cpu().numpy()
            dur_pred = dur_pred.squeeze().detach().cpu().numpy()
            dur_tgt = dur_tgt.squeeze().detach().cpu().numpy()
            pitch_pred = pitch_pred.squeeze().detach().cpu().numpy()
            pitch_tgt = pitch_tgt.squeeze().detach().cpu().numpy()
            energy_pred = energy_pred.squeeze().detach().cpu().numpy()
            energy_tgt = energy_tgt.squeeze().detach().cpu().numpy()
            if attn_soft is not None:
                attn_soft = attn_soft.squeeze().detach().cpu().numpy()
            if attn_hard is not None:
                attn_hard = attn_hard.squeeze().detach().cpu().numpy()

            for index in range(batch_size):
                if image_index == args.eval.max_num_images:
                    break
                input_length = in_lens[index]
                output_length = out_lens[index]
                mel_pred_figure = plot_spectrogram_to_figure(spectrogram=mel_pred[index, :, :output_length], info="Predicted Mel")
                mel_tgt_figure = plot_spectrogram_to_figure(spectrogram=mel_tgt[index, :, :output_length], info="Target Mel")
                dur_figure = plot_to_figure(predicted=dur_pred[index, :input_length], target=dur_tgt[index, :input_length], info="Duration")
                pitch_figure = plot_to_figure(predicted=pitch_pred[index, :input_length], target=pitch_tgt[index, :input_length], info="Pitch")
                energy_figure = plot_to_figure(predicted=energy_pred[index, :input_length], target=energy_tgt[index, :input_length], info="Energy")
                logger.log_tensorboard(tag=f"{image_index}/mel_pred", data=mel_pred_figure, global_step=args.current_train_step, type="figure")
                logger.log_tensorboard(tag=f"{image_index}/mel_tgt", data=mel_tgt_figure, global_step=args.current_train_step, type="figure")
                logger.log_tensorboard(tag=f"{image_index}/duration", data=dur_figure, global_step=args.current_train_step, type="figure")
                logger.log_tensorboard(tag=f"{image_index}/pitch", data=pitch_figure, global_step=args.current_train_step, type="figure")
                logger.log_tensorboard(tag=f"{image_index}/energy", data=energy_figure, global_step=args.current_train_step, type="figure")
                if attn_soft is not None:
                    attn_soft_figure = plot_alignment_to_figure(alignment=attn_soft[index, :output_length, :input_length])
                    logger.log_tensorboard(tag=f"{image_index}/attn_soft", data=attn_soft_figure, global_step=args.current_train_step, type="figure")
                if attn_hard is not None:
                    attn_hard_figure = plot_alignment_to_figure(alignment=attn_hard[index, :output_length, :input_length])
                    logger.log_tensorboard(tag=f"{image_index}/attn_hard", data=attn_hard_figure, global_step=args.current_train_step, type="figure")
                image_index += 1
        
    averager.update({'time': time.time() - args.last_log})
    averaged_stats = averager.average()
    log_stats_to_tensorboard(averaged_stats, logger, args, stage="eval")
    logger.log_stats(stats=averaged_stats, step=args.current_train_step, args=args, prefix='eval/')


def predict(model, teacher_model, dataloader, logger, args, vocoder_model, vocoder_hparams, accelerator):
    args.last_log = time.time()

    model = accelerator.unwrap_model(model)

    def mel_to_waveform(mel):
        mel = mel.permute(0, 2, 1)
        if vocoder_hparams.name == "causal_hifigan":
            waveform = vocoder_model(mel)
        else:
            raise NotImplementedError
        return waveform
    
    num_predict_samples = 0
    
    for batch_id, batch in enumerate(dataloader):
        model_outputs = forward(model, batch, args, teacher_model=teacher_model, predict=True)
        if args.mode == "pretrain":
            mel_tgt = batch[2].permute(0, 2, 1)
            mel_lens_tgt = batch[8]
            mel_out = model_outputs[0][0]
            dec_mask_pred = model_outputs[0][1]
        elif args.mode == "distill":
            mel_tgt = model_outputs[1][0][0]
            dec_mask_tgt = model_outputs[1][0][1]
            mel_lens_tgt = dec_mask_tgt.squeeze(2).int().sum(1)
            mel_out = model_outputs[0][0][0]
            dec_mask_pred = model_outputs[0][0][1]
        else:
            raise NotImplementedError
        
        wav_lens_tgt = mel_lens_tgt * vocoder_hparams.hop_size
        mel_lens_pred = dec_mask_pred.squeeze(2).int().sum(1)
        wav_lens_pred = mel_lens_pred * vocoder_hparams.hop_size
        wav_pred = mel_to_waveform(mel_out)
        wav_tgt = mel_to_waveform(mel_tgt)
        wav_pred = wav_pred.squeeze().detach().cpu().numpy()
        wav_tgt = wav_tgt.squeeze().detach().cpu().numpy()
        batch_size = mel_out.shape[0]
        
        for index in range(batch_size):
            if num_predict_samples >= args.infer.max_predict_samples:
                break
            predicted = wav_pred[index][:wav_lens_pred[index]]
            ground_truth = wav_tgt[index][:wav_lens_tgt[index]]
            logger.log_tensorboard(f"predicted/{num_predict_samples}", data=predicted, global_step=args.current_train_step, \
                type="audio", sample_rate=vocoder_hparams.sampling_rate)
            logger.log_tensorboard(f"ground_truth/{num_predict_samples}", data=ground_truth, global_step=args.current_train_step, \
                type="audio", sample_rate=vocoder_hparams.sampling_rate)
            num_predict_samples += 1
        
        if num_predict_samples >= args.infer.max_predict_samples:
            break
    
    logger.log_stats(
        stats={
            "time": time.time() - args.last_log,
        },
        step=args.current_train_step,
        args=args,
        prefix="test/",
    )


def maybe_grad_clip_and_grad_calc(accelerator, model, args):
    if args.optim.grad_clip > 0:
        grad_l2 = accelerator.clip_grad_norm_(
            parameters=model.parameters(),
            max_norm=args.optim.grad_clip,
            norm_type=2,
        )
    else:
        grad_l2 = None

    if args.logging.grad_l2:
        if grad_l2 is None:
            grad_l2 = (
                sum(p.grad.detach().data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5
            )

        return {'grad_l2': grad_l2}
    else:
        return {}


def extend_acc_stats(acc_stats, stats):
    for key in stats:
        if key not in acc_stats:
            acc_stats[key] = [stats[key]]
        elif isinstance(acc_stats[key], list):
            acc_stats[key].append(stats[key])
        else:
            acc_stats[key] = [acc_stats[key], stats[key]]

    return acc_stats


def average_stats(stats):
    for key in stats:
        if isinstance(stats[key], list):
            stats[key] = sum(stats[key]) / len(stats[key])
        else:
            stats[key] = stats[key].mean().item()

    return stats


def extra_stats(args, model, optimizer):
    stats = {}

    if args.logging.weights_l2:
        weights_l2 = sum(p.detach().norm(2).item() ** 2 for p in model.parameters()) ** 0.5
        stats['weights_l2'] = weights_l2

    stats['lr'] = optimizer.param_groups[0]['lr']
    stats['seconds_per_step'] = (time.time() - args.last_log) / args.logging.every_steps

    return stats


def maybe_logging(averager, args, model, optimizer, logger):
    if args.current_train_step % args.logging.every_steps == 0:
        stats = extra_stats(args, model, optimizer)

        averager.update(stats)
        averaged_stats = averager.average()

        logger.log_stats(
            stats=averaged_stats,
            step=args.current_train_step,
            args=args,
            prefix='train/'
        )

        args.last_log = time.time()


def log_stats_to_tensorboard(stats, logger, args, stage="train"):
    for key in stats:
        logger.log_tensorboard(
            tag=f"{stage}/{key}",
            data=stats[key],
            global_step=args.current_train_step,
            type="scalar"
        )


def forward(model, batch, args, teacher_model=None, predict=False):
    if args.mode == "pretrain":
        symbol_ids, num_symbols_of_each_word, mel, pitch, energy, \
            attention_prior, speaker_id, input_lengths, output_lengths = batch
        attention_prior = attention_prior_annealing(
            attention_prior=attention_prior,
            current_train_step=args.current_train_step,
            start_scale_down_step=args.optim.attn_prior_start_scale_down_step,
            end_step=args.optim.attn_prior_end_step
        )
    elif args.mode == "distill":
        symbol_ids, num_symbols_of_each_word, speaker_id, input_lengths = batch

    if args.mode == "pretrain":
        if predict:
            use_gt_pitch=False
            use_gt_energy=False
            inputs = (symbol_ids, input_lengths, num_symbols_of_each_word, speaker_id)
            text_only = True
        else:
            use_gt_pitch=True
            use_gt_energy=True
            inputs = (symbol_ids, input_lengths, num_symbols_of_each_word, mel, output_lengths, pitch, energy, attention_prior, speaker_id)
            text_only = False
        
        model_outputs = model(
            inputs=inputs, 
            use_gt_pitch=use_gt_pitch, 
            use_gt_energy=use_gt_energy,
            text_only=text_only
        )

    elif args.mode == "distill":
        with torch.no_grad():
            teacher_outputs = teacher_model.infer_teacher(inputs=symbol_ids, speaker=speaker_id)
            if not predict:
                teacher_dur_pred = teacher_outputs[0][2]
                teacher_pitch_pred = teacher_outputs[0][4]
                teacher_energy_pred = teacher_outputs[0][5]
            else:
                teacher_dur_pred = None
                teacher_pitch_pred = None
                teacher_energy_pred = None
                
        student_outputs = model(
            inputs=(symbol_ids, input_lengths, num_symbols_of_each_word, speaker_id),
            use_gt_pitch=False,
            use_gt_energy=False,
            dur_guide=teacher_dur_pred,
            pitch_guide=teacher_pitch_pred,
            energy_guide=teacher_energy_pred,
            text_only=True
        )

        model_outputs = (student_outputs, teacher_outputs)

    return model_outputs


def train(model, teacher_model, train_dataloader, dev_dataloader, accelerator, lr_scheduler, optimizer, loss_function, logger, args, vocoder_model, vocoder_hparams):
    model.train()
    train_averager = Averager()
    start_from = (args.current_train_step * args.optim.grad_acc) % len(train_dataloader)

    while args.current_train_step <= args.optim.total_steps:
        optimizer.zero_grad(set_to_none=True)
        acc_stats = {}

        for batch_id, batch in enumerate(train_dataloader, start=start_from):
            if args.current_train_step > args.optim.total_steps:
                break

            model_outputs = forward(model, batch, args, teacher_model)
            loss, stats = loss_function(model_outputs, batch)

            gathered_stats = accelerator.gather_for_metrics(stats)
            averaged_stats = average_stats(gathered_stats)
            acc_stats = extend_acc_stats(acc_stats, averaged_stats)
            accelerator.backward(loss / args.optim.grad_acc)
            train_averager.update(averaged_stats)

            if batch_id % args.optim.grad_acc == 0:
                stats = maybe_grad_clip_and_grad_calc(accelerator, model, args)
                gathered_stats = accelerator.gather_for_metrics(stats)
                # average stats across multiple gpus
                averaged_stats = average_stats(gathered_stats)
                acc_stats = extend_acc_stats(acc_stats, averaged_stats)
                # average stats across multiple grad acc steps
                averaged_acc_stats = average_stats(acc_stats)
                log_stats_to_tensorboard(averaged_acc_stats, logger, args, stage="train")
                acc_stats.clear()
                train_averager.update(averaged_stats)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                maybe_logging(train_averager, args, model, optimizer, logger)
                maybe_eval_predict(model, teacher_model, dev_dataloader, logger, args, loss_function, vocoder_model, vocoder_hparams, accelerator)
                maybe_save_checkpoint(accelerator, lr_scheduler, args)

                args.current_train_step += 1

        start_from = 1
    
    maybe_eval_predict(model, teacher_model, dev_dataloader, logger, args, loss_function, vocoder_model, vocoder_hparams, accelerator)
    maybe_save_checkpoint(accelerator, lr_scheduler, args)


@hydra.main(config_path="configs", config_name="instantspeech_ljspeech", version_base='1.1')
def main(args):
    accelerator = Accelerator(
        cpu=args.device == "cpu",
        mixed_precision=args.precision,
    )

    logger = setup_basics(accelerator, args)
    print(args)
    config = get_config(args)
    print(config)
    model, teacher_model = get_model(args, config)
    print(model)
    if teacher_model is not None:
        teacher_model = teacher_model.cuda()
    vocoder_model, vocoder_hparams = get_vocoder_model(args, config)
    vocoder_model = vocoder_model.cuda()
    print(vocoder_model)
    train_dataloader, dev_dataloader = get_dataloaders(args, config)
    print(args)
    optimizer = get_optimizer(model, args)
    print(optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, args)
    print(lr_scheduler)
    loss_function = get_loss_function(args)

    (
        model, optimizer, train_dataloader, dev_dataloader
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, dev_dataloader
    )

    with open_dict(args):
        args.current_train_step = 1
        args.last_log = time.time()

    if args.model.restore_from != "":
       accelerator.load_state(args.model.restore_from)
       scheduler_path = os.path.join(args.model.restore_from, "scheduler.bin")
       if os.path.exists(scheduler_path):
           lr_scheduler.load_state_dict(torch.load(scheduler_path))
       logger.log_message(f"Restored from {args.model.restore_from}")

       scheduler_state_dict = lr_scheduler.state_dict()
       with open_dict(args):
           args.current_train_step = scheduler_state_dict["last_epoch"] + 1

    if args.model.compile:
        model = torch.compile(model)

    train(model, teacher_model, train_dataloader, dev_dataloader, accelerator, lr_scheduler, optimizer, loss_function, logger, args, vocoder_model, vocoder_hparams)

    logger.finish()


if __name__ == "__main__":
    main()