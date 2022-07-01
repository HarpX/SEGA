###############################################################################################
#######                          Suivi des versions de notebook                         #######
#######                              CV_SEGA_model_objdetect                            #######
###############################################################################################




# OPTIMS :

Traiter le cas où j'ai plusieurs BB sur lamême image
Est-ce qu'il ne faut pas remélanger le dataset ?
Pour les landmarks / keypoints on utilise un algo qui détecte la zone puis qui donne les landmarks à partir de la zone à traiter.
Object detection -> Yolo est top
Attention, il semble y avoir des images dupliquées dans le dataset, regarder ça dans l'explo
Mise en place d'un LR sinus
intégrer un defreeze des couches pour pour spécifier le modèle sur notre application
ceuser la data augmentation (lum / expo, ...)
Travailler sur le type de backbone (Effnet XX, ResNet, DARKNet...)
Voir l'utilisation de decay dans le lr

Su NB v1 à 3 :
filtrer à une seule BB pour les premères itérations des NB
dupliquer les corrections de la fonction V_Flip, get_random_transform te magic_loader
tester d'autres variantes de back bones
dupliquer la modif du split


# CV_SEGA_model_objdetect_4 :
- intégration d'un modèle de type Yolo "simplifié"
- extraction du log des versions de notebook
- correction de la fonction de data augmentation qui ne retournait pas la valeur dans la bonne variable
- intégration de transformations affines
- révisionde l'approche de traitement es coordonnées de BB. On va utiliser x_moy et y_moy

# CV_SEGA_model_objdetect_3 :

    Intentions : Approche de prédiction d'une BB à partir d'une image ainsi que la probabilité de présence de cette BB. Seule une BB par image a été retenue pour le test.

- Révision de la loss et ajustement des coeff lamda

    Notes : les résultats ne sont pas bons. Le modèle prédit systématiquement les mêmes dimensions de BB. La probabilité de présence d'objet reste à 1 quoiqu'il arrive.

    # CV_SEGA_model_objdetect_3_1 :
    - utilisation de x_moy et y_moy
    - mise en place de data augmentation
    - Changement du backbone par un EfficientNetB7



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
    L'EfficientNetB7 a un impact positif sur ler performance du modèle.
    Le cosinus decay est à reconduire.
    Les changements d'optimizer n'ont pas eu d'impact.