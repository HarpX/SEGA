###############################################################################################
#######                          Suivi des versions de notebook                         #######
#######                              CV_SEGA_model_objdetect                            #######
#######                                 approche type Yolo                              #######
###############################################################################################




# OPTIMS :

Prioritaire :
Désactiver le filtrage des petites BB dans l'explo car on se retrouve avec des containers sans BB. Cela permettra d'améliorer la perfo
Refaire tourner la branche 4 avec le nouveau dataset
Sur la branche 5 :
    - Voir pour mettre en place des metriques
        https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
        https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/eager_few_shot_od_training_tf2_colab.ipynb
        l'AP, mAP ?
    - ajuster le nombre de classe en réduisant à 1 car il n'y en a qu'une
    - Voir pour activer la GPU sur colab
        with tf.device()
    - Voir pour sauvegarder l'entrainement
    - Travaller sur le learning rate
    - Que faire du set de stanford dans ce notebook, est-ce utile ?
    - quel est le type de modèle que j'utilise ?
    - voir pour utiliser plus de modèles
    - grossir la résolution ?
    - Voir pour faire tourner le notebook en local et non sur colab

Optionnel :
    Sur NB v1 à 2 :
    - Appliquer les corrections mise en place dans les NB de 1 à 2

# Choix des dimensions d'entrée en fonction du modèle
    EfficientNetB0 - (224, 224, 3)
    EfficientNetB1 - (240, 240, 3)
    EfficientNetB2 - (260, 260, 3)
    EfficientNetB3 - (300, 300, 3)
    EfficientNetB4 - (380, 380, 3)
    EfficientNetB5 - (456, 456, 3)
    EfficientNetB6 - (528, 528, 3)
    EfficientNetB7 - (600, 600, 3)

# CV_SEGA_model_objdetect_5 :

Intentions : Approche de prédiction de plusieurs BB à partir d'une image ainsi que la probabilité de présence de cette BB. Utilisatio d'un modèle tensorflow pré entrainné.

- Mise en application du notebook 8 du cours sur notre problèmatique
- La détection en multi BB fonctionne. L'entrainement sur un petit nombre d'epochs génère parfois des détections hazardeuses.
    # CV_SEGA_model_objdetect_5_1 :
    Epuration du notebook
    entrainement sur 5 epochs

# CV_SEGA_model_objdetect_4 :

    Intentions : Approche de prédiction de plusieurs BB à partir d'une image ainsi que la probabilité de présence de cette BB.

- intégration d'un modèle de type Yolo "simplifié"
- extraction du log des versions de notebook
- correction de la fonction de data augmentation qui ne retournait pas la valeur dans la bonne variable
- intégration de transformations affines
- révision de l'approche de traitement es coordonnées de BB. On va utiliser x_moy et y_moy

    # CV_SEGA_model_objdetect_4_1 :
    - mise à jour des fonction de chargements de dataset depuis le CV_SEGA_model_objdetect_3_1
    - Ajout du dataset de BG de stanford

    # CV_SEGA_model_objdetect_4_2 :
    Essai effectué avec une seule BB par image
    - Adaptation des sorties des générateurs pour donner des matrices 8x8
    - correction du split et suppr de la proba de présence
    - ajout d'une métrique sur la probabilité
    - ajout d'un learning rate cyclique
    - mise en place d'un optimizer de type SGD car l'ADAM restait bloqué

    # CV_SEGA_model_objdetect_4_3 :
    Essai effectué avec pluseurs BB par image
    - ajustement du dataset et mise en forme des données
    - travail sur la fonction d'intégration des différentes BB dans la matrice Yolo
    - désactivation de la partie data augmentation pour cet essai



# CV_SEGA_model_objdetect_3 :

    Intentions : Approche de prédiction d'une BB à partir d'une image ainsi que la probabilité de présence de cette BB. Seule une BB par image a été retenue pour le test.

- Révision de la loss et ajustement des coeff lamda

    Notes : les résultats ne sont pas bons. Le modèle prédit systématiquement les mêmes dimensions de BB. La probabilité de présence d'objet reste à 1 quoiqu'il arrive.

    # CV_SEGA_model_objdetect_3_1 :
    - changement du csv pour chargement afin de récupérer plus de données sur lesimages
    - utilisation de x_moy et y_moy
    - changement du train_test split
    - modification de la taille pour redimensionnement
    - mise en place de data augmentation
    - révision de la fonction de chargement, redimensionnement et data augmentation
    - ajustement des générateurs
    - Changement du backbone par un EfficientNetB2

    Perfos après un entrainement de 40 epochs :
        loss: 0.1023 - metric_iou: 0.6971 - metric_confident: 1.0000

    Perfos après un défreeze de 5 couches et entrainement de 40 epochs :
        loss: 0.0967 - metric_iou: 0.7235 - metric_confident: 1.0000 - val_loss: 0.0361 - val_metric_iou: 0.7483 - val_metric_confident: 1.0000

    # Conclusions :
    Le problème de constance des coordonnées de BB est à présent réglé.
    Les performances sont correctes mais on détecte quand même des bennes sur des images qui n'en n'ont pas.
    Cette technique ne permet de détecter qu'une seule benne sur l'image.

# CV_SEGA_model_objdetect_2 :

    Intentions : Approche de prédiction d'une BB à partir d'une image ainsi que la probabilité de présence de cette BB.

- Ajout de la data augmentation dans la fonction de chargement
- intégration du set de données stanford background pour fournir une probabilité de détection d'objet en sortie
- modification du générateur
- correction du problème de normalisation des coordonnées dans la fonction de chargement du set de bennes
- construction des loss et des metrics
- correction du générateur de test
- ajout de courbes sur la partie affichage des paramètre d'entrainement
- mise en place de la partie inférence.
- début de codage du defreeze

    Notes : les résultats ne sont pas bons. Le modèle prédit systématiquement les mêmes dimensions de BB

# CV_SEGA_model_objdetect_1 :

    Intentions : Approche de prédiction d'une BB à partir d'une image. Seule une BB par image a été retenue pour le test.

- création du notebook
- développement de la fonction de chargement et de redimensionnement
- première modélisation simple pour la prédiction d'une seule BB par image
- ajout des callbacks

    # CV_SEGA_model_objdetect_1_1
    - suppresion des cas avec plus d'une BB pour tester
    - ajout de la partie inférence

     Notes : les résultats ne sont pas bons. Le modèle prédit systématiquement les mêmes dimensions de BB.
     loss: 0.1228 - mean_squared_error: 0.4604

    # CV_SEGA_model_objdetect_1_2
    EfficientNetB0 + Changement du learning rate. Ajout d'un scheduler cosinus decay

    loss: 0.1260 - mean_squared_error: 0.0493

    # CV_SEGA_model_objdetect_1_3
    EfficientNetB0 + Changement du learning rate. Ajout d'un scheduler CyclicalLearningRate

    loss: 0.1237 - mean_squared_error: 0.5499

    # CV_SEGA_model_objdetect_1_4
    Changement du backbone par un EfficientNetB7

    loss: 0.1231 - mean_squared_error: 0.0478

    # CV_SEGA_model_objdetect_1_5
    Changement du backbone par un ResNet50

    loss: 0.1231 - mean_squared_error: 0.4582

    # CV_SEGA_model_objdetect_1_6
    Changement du backbone par un ResNet50 + SDG en optimizer

    loss: 0.1219 - mean_squared_error: 0.5512

    # CV_SEGA_model_objdetect_1_7
    Changement du backbone par un ResNet50 + RMSprop en optimizer

    loss: 0.1282 - mean_squared_error: 0.9660

    # Conclusions :
    Le problème de constance des coordonnées de BB est toujours présent. Une erreur doit se trouver au niveau de la partie chargement.
    L'EfficientNetB7 a un impact positif sur ler performance du modèle.
    Le cosinus decay est à reconduire.
    Les changements d'optimizer n'ont pas eu d'impact.



