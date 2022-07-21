###############################################################################################
#######                          Suivi des versions de notebook                         #######
#######                              CV_SEGA_model_keypoints                            #######
#######                                        --                                       #######
###############################################################################################

# Intentions :
Prédiction de 7 à 8 keypoints entourrant un container.
Un seul container par image sera retenu.


# OPTIMS :

Prioritaire :

Optionnel :


# Choix des dimensions d'entrée en fonction du modèle
    EfficientNetB0 - (224, 224, 3)
    EfficientNetB1 - (240, 240, 3)
    EfficientNetB2 - (260, 260, 3)
    EfficientNetB3 - (300, 300, 3)
    EfficientNetB4 - (380, 380, 3)
    EfficientNetB5 - (456, 456, 3)
    EfficientNetB6 - (528, 528, 3)
    EfficientNetB7 - (600, 600, 3)

# CV_SEGA_model_keypoints_1_1 :

    Intentions : création d'une architecture permettant de répondre à la problématique en attendant les données finales.

    - création du notebook
    - On réutilise le dataset des container en le filtrant sur des BB dont l'aire est comprise entre 25 et 75 % de l'image. On limitera aussi les images à un container comme défini lors de notre dernière réunion de projet.
    - utilisation du EfficientNetB5
    - utilisation d'une MAE pour la loss

    loss: 0.1857 - mean_absolute_error: 0.1857 - val_loss: 0.1232 - val_mean_absolute_error: 0.1232

# CV_SEGA_model_keypoints_1_1 :

    Intentions : création d'une architecture permettant de répondre à la problématique en attendant les données finales.

    - /255 de l'image

    loss: 0.1325 - mean_absolute_error: 0.1325 - val_loss: 0.0912 - val_mean_absolute_error: 0.0912


# CV_SEGA_model_keypoints_2_1 :

    Intentions : tester l'architecture (EfficientNetB5) précédemment créée sur un dataset provisoire.

    - adaptation du code pour lire le nouveau dataset provenant de VIA
    - normalisation des coordonnées pour le travail
    - Suppression de la division par 255 lors du chargement
    - Désactivation de la data augmentation dans un premier temps
    - adaptation du modèle pour sortir 4 keypoints
    - sortie d'activation en sigmoid

    loss: 0.2074 - mean_absolute_error: 0.2074 - val_loss: 0.2227 - val_mean_absolute_error: 0.2227

    Le positionnement des coordonnées n'est pas assez précis. A tester avec le vrai dataset ainsi qu'une data augmentation.


# CV_SEGA_model_keypoints_2_2 :

    Intentions : tester l'architecture (EfficientNetB5) précédemment créée sur un dataset provisoire. 

    - Suppression de la division par 255 lors du chargement
    - modification de la sortie d'activation en linear

    loss: 0.2200 - mean_absolute_error: 0.2200 - val_loss: 0.1936 - val_mean_absolute_error: 0.1936


# CV_SEGA_model_keypoints_2_3 :

    Intentions : tester l'architecture (EfficientNetB5) précédemment créée sur un dataset provisoire. 

    - ajout de la partie data augm





tester le VGG16

forçage installé dans le 3_1 pour le dev de la partir dataaug

Reprendre es notes de cahier
travailler sur de la data augmentation
essayer dataug
Essayer albumentation par la suite https://albumentations.ai/
Pourquoi on n'utiliserai pas un GAN avec un unet ?
intégrer le gradcam ou feature map
Travailler sur une loss adaptée
Construire des métriques



    # Conclusions :
    ...



