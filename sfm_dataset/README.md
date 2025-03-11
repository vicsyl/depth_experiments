# SFM dataset


## Basic workflow:

### Find out what to download

* dataset_stats.py (very heavy-weight, not recommended) - to get some idea about dataset stats
* lightweight.py (recommended) - to get raw dataset stats - see ../data/lightweight_stats.txt
  * TODO idea - just really download and handle per reconstruction?
  * e.g., first entries with largest size of images.bin (for all reconstructions all together)
``` 
'002/236' 'Dom_Luis_I_bridge'
'028/023' 'Sagrada_Família'
'025/512' 'Cathédrale_Notre-Dame_de_Paris'
'177/780' 'Alhambra'
``` 

### Download the data:

* e.g. ~/dev/datasets/megascenes$ ./run_download.sh '025/041' 'Santa_Maria_del_Fiore_Florence'


### Add the data to SfM dataset

* e.g. 

```
python -u depths_from_sfm.py \ 
      --write \
      --no-only_info \
      --scenes new \
      --megascenes_data_path '../datasets/megascenes' \
       --marigold_rel_path '../marigold_private'"

```

### Generate the split

### Use the dataset for training



