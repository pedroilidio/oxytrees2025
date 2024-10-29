FOLDERS=(
    "prediction_weights"
    "bipartite_adaptations"
    "y_reconstruction"
    "semisupervised_forests"
    "best_forests_with_dropout"
    "literature_models"
)

for D in "${FOLDERS[@]}" ; do
    bash $D/plot.sh
done
