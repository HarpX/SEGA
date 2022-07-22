###############################################################################################
#######                          Suivi des versions de notebook                         #######
#######                              CV_SEGA_model_keypoints                            #######
#######                                        --                                       #######
###############################################################################################

# Intentions :
Prédiction de 4 keypoints entourrant un container.
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

==================================================================================================
# CV_SEGA_model_keypoints_1 :
## CV_SEGA_model_keypoints_1_1 :

    Intentions : création d'une architecture permettant de répondre à la problématique en attendant les données finales.

    - création du notebook
    - On réutilise le dataset des container en le filtrant sur des BB dont l'aire est comprise entre 25 et 75 % de l'image. On limitera aussi les images à un container comme défini lors de notre dernière réunion de projet.
    - utilisation du EfficientNetB5
    - utilisation d'une MAE pour la loss

    loss: 0.1857 - mean_absolute_error: 0.1857 - val_loss: 0.1232 - val_mean_absolute_error: 0.1232

## CV_SEGA_model_keypoints_1_2 :

    Intentions : création d'une architecture permettant de répondre à la problématique en attendant les données finales.

    - /255 de l'image

    loss: 0.1325 - mean_absolute_error: 0.1325 - val_loss: 0.0912 - val_mean_absolute_error: 0.0912

==================================================================================================

# CV_SEGA_model_keypoints_2 :

## CV_SEGA_model_keypoints_2_1 :

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

## Conclusions :
Cette première base d'architecture est prête pour être le point de départ de la suite du développement.
Pour le moment, elle ne prédit que des coordonnées de bounding box donc à adapter.


## CV_SEGA_model_keypoints_2_3 :

    Intentions : tester l'architecture (EfficientNetB5) précédemment créée sur un dataset provisoire. 

    - ajout de la partie data augm (maison)

    loss: 0.2270 - mean_absolute_error: 0.2270 - val_loss: 0.1749 - val_mean_absolute_error: 0.1749

## CV_SEGA_model_keypoints_2_4 :

    Intentions : même notebook que les précédent mais avec un backbone VGG16

    - modification de la partie encoder par un back bone en VGG13=6

    loss: 0.3172 - mean_absolute_error: 0.3172 - val_loss: 0.2598 - val_mean_absolute_error: 0.2598

## Conclusions :
Les localisations ne sont pas efficaces. La loss stagne assez rapidement et il n'y a pas de gain de performance.
La meilleure configuration pour le moment est celle du 2_3 avec un EfficientNetB5 et de la data aug custom.
Il sera certainement nécessaire de tester une data augmentation plus performante ainsi que le vrai dataset.
Les images qui nous ont été 

==================================================================================================

# CV_SEGA_model_keypoints_3 :

    Intentions : Prédire des keypoints qui constitueront un parallélépipède autour d'un container avec des images qui ne contiennent qu'un container.
    Étant donnée qu'un parallélépipède est défini par 4 points et un sommet, nous n'allons travailler que sur la prédiction de 4 points. 
    Les autres points seront déterminés par reconstruction géométrique via des fonctions.
    Un micro dataset a été créé afin de tester l'approche en attendant le dataset définitif.
    Dans cette version de notebook, nous allons tenter d'ajuster l'agorithme mmpose de openMMlab à notre problématique de containers.

    Liens : 
    - https://github.com/open-mmlab/mmpose
    - https://mmpose.readthedocs.io/en/latest

## CV_SEGA_model_keypoints_3_1 :

    Intentions : Premiers pas avec mmpose

    - Application de la doc : https://github.com/open-mmlab/mmpose/blob/master/docs/en/install.md
    - Prerequisites : Step 1 + Step 2 case CPU
    - Best Practices : step 0 + step 1 case b
    - Verify the installation : en totalité sur colab
    - mise en forme des fichiers et dossiers en fonction de ce qui est attendu par mmpose -> Il y a un point de bloquage à discuter avec Thomas






Reprendre mes notes de cahier
Voir ce que donne le json d'Akanthas

essayer dataug
Essayer albumentation par la suite https://albumentations.ai/

approches "classiques" puis je vais tenter un genre de Yolo masterisé 

Pourquoi on n'utiliserai pas un GAN avec un unet ?

intégrer le gradcam ou feature map
Travailler sur une loss adaptée
Construire des métriques

    # Conclusions :
    ...



