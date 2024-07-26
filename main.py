from generate_masks import process_files


# Define your paths here
svs_folder = "/your/original svs images/folder"
geojson_folder = "/your/geojson data/folder"
output_folder = "/store binary masks here/folder"

# Process SVS files and corresponding GeoJSON files
process_files(svs_folder, geojson_folder, output_folder)


