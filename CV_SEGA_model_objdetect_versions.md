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
- intégration d'un modèle de type Yolo "simple"
- extraction du log des versions de notebook
- correction de la fonction de data augmentation qui ne retournait pas la valeur dans la bonne variable
- intégration de transformations affines

# CV_SEGA_model_objdetect_3 :
- Révision de la loss et ajustement des coeff lamda

    Notes : les résultats ne sont pas bons. Le modèle prédit systématiquement les mêmes dimensions de BB. La probabilité de présence d'objet reste à 1 quoiqu'il arrive.

# CV_SEGA_model_objdetect_2 :
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
- création du notebook
- développement de la fonction de chargement et de redimensionnement
- première modélisation simple pour la prédiction d'une seule BB par image
- ajout des callbacks
