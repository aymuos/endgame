# endgame
Final repository for code for masters thesis 



---
##

Invalid notebook fix 

# Create a backup first (optional but recommended)
```cp notebook.ipynb notebook_backup.ipynb```

### Use jq to delete the metadata.widgets key and save to a temporary file
`jq 'del(.metadata.widgets)' notebook.ipynb > temp_notebook.ipynb`

### Replace the original file with the fixed one
```mv temp_notebook.ipynb notebook.ipynb```