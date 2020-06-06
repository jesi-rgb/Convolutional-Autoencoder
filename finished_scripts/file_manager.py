import os
import json
import pandas as pd

dataset_info_path = "data/dataset_info.json"
locations_path = "data/locations"
cct_path = "data/cct_images/"


# cargamos el dataset. Esto se guarda como un dict() de python.
dataset_info = None
with open(dataset_info_path) as json_file:
    dataset_info = json.load(json_file)

# extraemos la información que nos interesa en dataframes, que son algo más convenientes para 
# consultar los datos pertinentes.
categories_dataframe = pd.DataFrame(data=dataset_info['categories'])
annotations_dataframe = pd.DataFrame(data=dataset_info['annotations'])
images_dataframe = pd.DataFrame(data=dataset_info['images'])

# buscamos cual es la etiqueta que indica que una imagen está vacía en el campo categories
empty_id = categories_dataframe[categories_dataframe['name'] == 'empty'].iloc[0]['id']

# buscamos el id de aquellas imágenes que estén etiquetadas como vacías en el campo annotations
empty_imgs_id = annotations_dataframe[annotations_dataframe['category_id'] != empty_id]['image_id'].to_list()

# creamos un conjunto con las localizaciones y un subdataframe de aquellas imágenes
locations_set = set(images_dataframe['location'])

# ahora sí, finalmente buscamos el id real (nombre de archivo) de aquellas imágenes que están vacías (el campo annotations
# y el campo images están relacionados por el atributo 'image_id' e 'id')
empty_imgs_df = images_dataframe[images_dataframe['id'].isin(empty_imgs_id)]

# creamos una carpeta para cada localización
for loc in locations_set:
    os.makedirs(os.path.join(locations_path, str(loc), 'animals'), exist_ok=True)

# empty_imgs_df es un dict con varios campos, así que cogemos los que nos interesan: location e id (id es el nombre de los archivos)
locations_df = empty_imgs_df['location'].to_list()
images_id_df = empty_imgs_df['id'].to_list()

# creamos un dict propio para enlazar ambas listas paralelamente
loc_img_dict = dict(zip(images_id_df, locations_df))

# creamos una lista de absolutamente todas las imágenes que tenemos
filenames = os.listdir(cct_path)

# para cada imagen del conjunto total
for file in filenames:
    # comprobamos que está en el subconjunto de imágenes vacías
    if file[:-4] in images_id_df:
        # si es así, obtenemos su localización
        loc = loc_img_dict[file[:-4]]
        
        # creamos las rutas de origen y destino
        src = os.path.join(cct_path, file)
        dst = os.path.join(locations_path, str(loc), 'animals', file)
        
        # movemos la imagen a la carpeta cuyo nombre es igual a la localización de la imagen
        os.rename(src, dst)