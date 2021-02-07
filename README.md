## Code and data for ACL submission: Learning Zero-shot Text Classification from Wikipedia
## Setup the environment
xxx
## Using our models
Users can download our pre-trained ESA-WikiCate and TE-WikiCate models ([model.zip](http://...)), and evaluation data ([data.zip](http://...)). 
To evaluate ESA_WikiCate, for example, run
```
python src/eval_ESA.py --data_path data/yahoo/test.txt --label_path data/yahoo/label_names.txt --esa_path model/ESA_WikiCate --n_jobs N_JOBS
```
To evaluate TE-WikiCate, for example, run
```
CUDA_VISIBLE_DEVICES=xxx python src/eval_TE.py --data_path data/yahoo/test.txt --label_path data/yahoo/label_names.txt --model_path model/TE_WikiCate/ --simple_hypo
```
For the multi-labeled *Situation* dataset, add ```--multi_label``` to the above two commands. 
## Training
We provide the data for training TE-WikiCate ([wikipedia.zip](http://...)), which is about 12GB. To train your own model, extract the zip file under ```data``` directory and run
```
CUDA_VISIBLE_DEVICES=xxx python src/train_wiki_entailment.py --save_dir SAVE_DIR
```



