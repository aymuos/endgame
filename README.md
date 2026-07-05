# endgame
Final repository for code for masters thesis 


# Sequence of notebooks 

- Run 06_weather.ipynb to generate weather files 
- run 00_d5c_seperator.ipynb to pull the code that runs the city seperated delivery files 
- Run 01_master_feature_creator.ipynb to generate features at batch level and delivery level 
- Run 02_PC_delviery_level.ipynb to run PC algorithm at each individual city level
- Run 03_PCMCI.ipynb to run pcmci at batch level : This confirms that batches are memory-less process 
- Run 04_cdnod.ipynb to run the cdnod algorithm from 3 cities at 1000 samples each . This is a long running script - runs for about 18 hours due to N3 complexity 



---
##

Invalid notebook fix 

# Create a backup first (optional but recommended)
```cp notebook.ipynb notebook_backup.ipynb```

### Use jq to delete the metadata.widgets key and save to a temporary file
`jq 'del(.metadata.widgets)' notebook.ipynb > temp_notebook.ipynb`

### Replace the original file with the fixed one
```mv temp_notebook.ipynb notebook.ipynb```