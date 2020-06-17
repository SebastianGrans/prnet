# Partial Registration Network
Fork of PRNet 

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install torch numpy scipy h5py tqdm sklearn
```

## Train
```
python3 main.py --exp_name "demo" --svd_on_gpu 
```

## Test
This will load a single sample with the same dataloader as the original author has written.
It then applies `PRNet.predict()` which runs the network three times on the source and traget network.

```
python3 main.py --exp_name "demo" --eval 
```

The following will also visualize the point clouds before and after transformation with the predicted R and t.
```
python3 main.py --exp_name "demo" --eval --visualize 
```


## Output from `pip list`

(Seems like I've accidentally installed some other things other than the aforementioned requirements.)

```
pip list --format columns
Package            Version                
------------------ -----------------------
attrs              19.3.0                 
backcall           0.1.0                  
decorator          4.4.2                  
h5py               2.10.0                 
importlib-metadata 1.6.0                  
ipywidgets         7.5.1                  
jedi               0.16.0                 
joblib             0.14.1                 
jsonschema         3.2.0                  
numpy              1.18.2                 
open3d             0.9.0.0                
parso              0.6.2                  
pexpect            4.8.0                  
pickleshare        0.7.5                  
Pillow             7.0.0                  
pip                20.0.2                 
pip-autoremove     0.9.1                  
pkg-resources      0.0.0                  
prompt-toolkit     3.0.5                  
ptyprocess         0.6.0                  
Pygments           2.6.1                  
pyrsistent         0.16.0                 
python-dateutil    2.8.1                  
pyzmq              19.0.0                 
scikit-learn       0.22.2.post1           
scipy              1.4.1                  
setuptools         39.0.1                 
six                1.14.0                 
sklearn            0.0                    
torch              1.5.0                  
torchvision        0.6.0.dev20200327+cu101
tornado            6.0.4                  
tqdm               4.43.0                 
traitlets          4.3.3                  
wcwidth            0.1.9                  
widgetsnbextension 3.5.1                  
zipp               3.1.0     
```



## Citation
Please cite this paper if you want to use it in your work,

	@InProceedings{Wang_2019_NeurIPS,
	  title={PRNet: Self-Supervised Learning for Partial-to-Partial Registration},
	  author={Wang, Yue and Solomon, Justin M.},
	  booktitle = {33rd Conference on Neural Information Processing Systems (To appear)},
	  year={2019}
	}


## License
MIT License
