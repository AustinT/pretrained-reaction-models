# Script to download retro-star files
file_name="retro_data.zip"
wget https://www.dropbox.com/s/ar9cupb18hv96gj/retro_data.zip?dl=0 -O ./files/$file_name
cd ./files
unzip $file_name

# Create canonical origin dict
cd dataset
python canonicalize_origin_dict.py
