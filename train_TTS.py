from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from ETTS.trainer import ETTSTrasnformerModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--saving_path', type=str, default='./checkpoints')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--training_step', type=int, default=250000)
parser.add_argument('--warmup_step', type=int, default=100)
parser.add_argument('--maxlength', type=int, default=2000)
parser.add_argument('--nmels', type=int, default=80)
parser.add_argument('--emo_embed_dim', type=int, default=768)
parser.add_argument('--text_embed_dim', type=int, default=256)
parser.add_argument('--model_dim', type=int, default=256)
parser.add_argument('--model_hidden_size', type=int, default=512*2)
parser.add_argument('--nlayers', type=int, default=5)
parser.add_argument('--nheads', type=int, default=2)
parser.add_argument('--ngst', type=int, default=64)
parser.add_argument('--nlst', type=int, default=32)
parser.add_argument('--resume_checkpoint', type=str, default=None)
parser.add_argument('--precision', type=int, choices=[16, 32], default=32)
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--accelerator', type=str, default='ddp')
parser.add_argument('--train_bucket_size', type=int, default=50)
parser.add_argument('--val_bucket_size', type=int, default=20)
parser.add_argument('--val_split', type=float, default=0.9)
parser.add_argument('--gate_pos_weight', type=float, default=6.0)
parser.add_argument('--nworkers', type=int, default=8)
parser.add_argument('--num_audio_sample', type=int, default=8)
parser.add_argument('--n_attn_plots', type=int, default=3)
parser.add_argument('--use_guided_attn', action='store_true')
parser.add_argument('--disable_lst', action='store_true')
parser.add_argument('--n_guided_steps', type=int, default=250000)
parser.add_argument('--etts_checkpoint', type=str, default=None)
parser.add_argument('--check_val_every_n_epoch', type=int, default=20)

parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--datatype', type=str, required=True, choices=['LJSpeech', 'VCTK'])
parser.add_argument('--vocoder_ckpt_path', type=str, required=True)
parser.add_argument('--sampledir', type=str, required=True)
args = parser.parse_args()

checkpoint_callback = ModelCheckpoint(
    dirpath=args.saving_path,
    filename='{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    verbose=True,
    monitor='val_loss',
    mode='min',
    save_last=True
)
wrapper = Trainer(
    precision=args.precision,
    amp_backend='native',
    callbacks=[checkpoint_callback],
    resume_from_checkpoint=args.resume_checkpoint,
    check_val_every_n_epoch=args.check_val_every_n_epoch,
    num_sanity_val_steps=10, #Disable sanity check
    max_steps=args.training_step,
    gpus=(-1 if args.distributed else 1),
    accelerator=(args.accelerator if args.distributed else None),
    replace_sampler_ddp=False
)

model = ETTSTrasnformerModel(datadir=args.datadir,
                             datatype=args.datatype,
                             vocoder_ckpt_path=args.vocoder_ckpt_path,
                             maxlength=args.maxlength,
                             nmels=args.nmels,
                             emo_embed_dim=args.emo_embed_dim,
                             text_embed_dim=args.text_embed_dim,
                             model_dim=args.model_dim,
                             model_hidden_size=args.model_hidden_size,
                             nlayers=args.nlayers,
                             nheads=args.nheads,
                             ngst=args.ngst,
                             nlst=args.nlst,
                             use_lst=(not args.disable_lst),
                             train_bucket_size=args.train_bucket_size,
                             val_bucket_size=args.val_bucket_size,
                             warmup_step=args.warmup_step,
                             maxstep=args.training_step,
                             batch_size=args.batch_size,
                             lr=args.lr,
                             distributed=args.distributed,
                             val_split=args.val_split,
                             gate_pos_weight=args.gate_pos_weight,
                             nworkers=args.nworkers,
                             num_audio_sample=args.num_audio_sample,
                             sampledir=args.sampledir,
                             n_attn_plots=args.n_attn_plots,
                             use_guided_attn=args.use_guided_attn,
                             n_guided_steps=args.n_guided_steps,
                             etts_checkpoint=args.etts_checkpoint)
wrapper.fit(model)
