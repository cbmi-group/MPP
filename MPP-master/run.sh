python ./dataset/_augment.py --stage surpevised
python train.py --stage supervised --epoch 100
python inference.py --stage supervised
python image_monography.py
python ./dataset/_augment.py --stage semi-surpevised
python train.py --stage semi-supervised --epoch 100
python test_Unet.py
python test_space_Unet.py