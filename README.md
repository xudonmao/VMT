# VMT

Code for [Virtual Mixup Training for Unsupervised Domain Adaptation](https://arxiv.org/abs/1905.04215).


### Acknowledgments

This code is based on [dirt-t](https://github.com/RuiShu/dirt-t).

### Dependencies

```python
numpy==1.14.1
scikit_image==0.13.1
scipy==1.0.0
tensorflow_gpu==1.6.0
tensorbayes==0.4.0
```

### Train

1. Run VMT
```
python -u run_dirtt.py --datadir data --src mnist --trg svhn --inorm 1 --run 0 --dirt 0 --dw 0.01 --svw 1 --tvw 0.06 --tcw 0.06 --smw 1 --tmw 0.06
```

2. Run DIRT-T 
```
python -u run_dirtt.py --datadir data --src mnist --trg svhn --inorm 1 --run 0 --dirt 5000 --dw 0.01 --svw 1 --tvw 0.06 --tcw 0.06 --smw 1 --tmw 0.06
```


### Citation
If you use this work in your research, please cite:

    @article{arxiv1905.04215,
      author = {Xudong Mao and Yun Ma and Zhenguo Yang and Yangbin Chen and Qing Li},
      title = {Virtual Mixup Training for Unsupervised Domain Adaptation},
      journal = {arXiv preprint arXiv:1905.04215},
      year = {2019}
    }
