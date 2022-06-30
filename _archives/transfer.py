################ Réalisation des sets d'entrainnement et de tests
"""
    taille du set de test => 10 % des données restantes
    random_state => 42
"""

# Séparer la variable explicative de la variable à prédire
X, y = df["file_name"], df["bbox"]

X_train, X_test, y_train, y_test  = train_test_split(np.array(X), np.array(y), test_size=0.1,random_state = 42)


===================


def get_random_transform(X,y):

    def V_Flip(X,y):
        X = tf.image.flip_left_right(X)
        x_up = X.shape[1] - y[0]- y[2]
        new_y = tf.stack([x_up,y[1],y[2], y[3]],0)
        return X,new_y

    def Identity(X,y):
        return X,y
        
    def img_slider(X, y):
        
        # Find the max translation to keep the object in the image
        tx_max = y[0]
        tx_min = - (X.shape[1] - y[0] - y[2])
        ty_max = y[1] 
        ty_min = -(X.shape[0] - y[1]-y[3])

        # Choose a andom translation
        tx = np.random.uniform(tx_min.numpy(), tx_max.numpy())
        ty = np.random.uniform(ty_min.numpy(), ty_max.numpy())

        # Apply the transformation in the image
        X_new = tf.keras.preprocessing.image.apply_affine_transform(X.numpy(),theta=0,
                                                                 tx=tx,ty=ty,shear=0,
                                                                 zx=1,zy=1,row_axis=0,
                                                                 col_axis=1,channel_axis=2,
                                                                 fill_mode='nearest',cval=0.0,
                                                                 order=1
                                                                )
        # Correct the target variable
        y_new = y.numpy()

        y_new[0] += -tx
        y_new[1] += -ty
        
        X = X_new
        y = y_new

        return X,y      


    # Génération aléatoire des transformations      
    p = tf.random.uniform(shape = [1], minval=0, maxval=3, dtype=tf.dtypes.int32)
    
    # Boucle de sélection
    if p == 0:
        X,y = Identity(X, y)
    elif p==1:
        X,y = V_Flip(X, y)
    elif p==2:
        X,y = img_slider(X, y)

    return X,y


# Fonction de chargement des images via un numpy array ou un chemin

def magic_loader (X, y, fromfile = False, norm = True, resize = None, dataAugmentation = False):   
    
    # ==== Chargement de l'image
    if fromfile == True:
        image = tf.io.read_file(X, name=None)
        image = tf.io.decode_image(image,channels=3,expand_animations=False)     
     
    else:
        image = tf.convert_to_tensor(X)
   
    # ==== Redimensionnement  
    if resize!= None:
        # Récupération de l'ancienne taille de l'image
        old_size = tf.shape(image)[:2]

        # Définition de la taille cible de l'image
        img_size = tf.constant(resize)

        # Redimensionnement
        image = tf.image.resize(image,
                             img_size,
                             method='nearest',
                             preserve_aspect_ratio=False,
                             antialias=False,
                             name=None)
        
        # Calcul du ratio de redimensionnement pour adapter les bounding boxs
        ratio = tf.math.divide(img_size,old_size)
        ratio = tf.cast(ratio, dtype="float32")
        

        
        # Redimensionnement des bounding boxs
        y = tf.convert_to_tensor(y)
        y = tf.cast(y, dtype="float32")
        x_sized = y[0]*ratio[1]
        y_sized = y[1]*ratio[0]
        w_sized = y[2]*ratio[1]
        h_sized = y[3]*ratio[0]
        y_new = tf.stack([x_sized,y_sized,w_sized,h_sized],0)

    else:
        y_new = y
    
    # ==== dataAugmentation 
    if dataAugmentation :
        image,y_new = get_random_transform(image,y_new)          
    
    # ==== Normalisation    
    if norm :
        image = tf.divide(image, 255)
        
    # ==== Ajout de la probabilité de présence d'objet
    pobj = tf.constant([1], dtype="float32" )
    y_new = tf.concat([pobj,y_new], axis = 0)
    
    return image, y_new 

# ================================================================================================
# Définition de l'image pour test de la fonction de chargement d'image
i = random.choices(list(df.index), k=1)[0]
img_path = df.loc[i,"file_name"]

# ================================================================================================
# Test de la fonction
image = plt.imread(img_path)

print("==== Avant ====")
print(image.shape)
print(df.loc[i,"bbox"])
show_img_bb (image, df.loc[i,"bbox"][0], df.loc[i,"bbox"][1], df.loc[i,"bbox"][2], df.loc[i,"bbox"][3], df.loc[i,"file_name"] )
plt.show()

# Application de la fonction fonctions
image_new, y_new = magic_loader(img_path, df.loc[i,"bbox"],
                                fromfile = True,
                                norm = True,
                                resize = (imgRsize[0], imgRsize[0]),
                                dataAugmentation = True)

print("==== Après ====")
print(image_new.shape)
print(y_new)
show_img_bb (image_new, y_new[1], y_new[2], y_new[3], y_new[4], df.loc[i,"file_name"] )
plt.show()