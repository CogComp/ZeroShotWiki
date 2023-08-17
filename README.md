## Code and data for NAACL 2022 Demo: Towards Open Domain Topic Classification
## Requirements
To install required Python packages, run
```
pip install -r requirement.txt
```
We recommend to create a new conda environment of Python 3.7 to avoid potential incompatibility.
## Using our models
*To download datasets/models on this page, please right-click on the links to "Save link as ..."*

Users can download our pre-trained ESA-WikiCate and TE-WikiCate models ([model.zip](http://cogcomp.org/models/ZeroShotWiki/model.zip)), and evaluation data ([data.zip](http://cogcomp.org/models/ZeroShotWiki/data.zip)). 
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
We provide the data for training TE-WikiCate ([wikipedia.zip](http://cogcomp.org/models/ZeroShotWiki/wikipedia.zip)), which is about 12GB. To train your own model, extract the zip file under ```data``` directory and run
```
CUDA_VISIBLE_DEVICES=xxx python src/train_wiki_entailment.py --data_file data/wikipedia/tokenized_wiki.txt --cate_file data/wikipedia/page2content_cate.json --save_dir SAVE_DIR
```




