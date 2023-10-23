ulimit -n 16384
python -W ignore scripts/train_lightning.py --config=config/multimodal_context_AQGT-A.yml
