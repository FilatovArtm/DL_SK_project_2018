To start the training run the following command

```
python3 train_pix2pix.py --name deep_fasion_test --model pix2pix --which_model_netG unet_128 --which_direction AtoB --lambda_A 100 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0
```

For inference run

```
python3 save_example.py --name deep_fasion --model pix2pix --which_model_netG unet_128 --which_direction AtoB --lambda_A 100 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0
```
