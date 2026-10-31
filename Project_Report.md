# Project 4: Breast Cancer Molecular Subtype Classification
Student: Shreya | Date: 31-Oct-2026

## Objective
Classify 4 subtypes (Luminal A, Luminal B, HER2-enriched, Triple-Negative) for correct treatment.

## Workflow
Collect Data -> Clean -> Split Train/Test -> Train Model -> Test Model -> Predict Subtype -> Check Accuracy

## Dataset
1000 patients, 8 raw + 6 engineered features, Target: Subtype
Logic: TN = ER-/PR-/HER2-, HER2-enriched = HER2+/ER-, Luminal A = ER+ & Ki67<20 & Grade<=2

## Methodology
- Cleaning: dtypes, missing, duplicates, IQR outliers
- Visuals: Subtype count, Ki67 boxplot, Tumor Size boxplot, Correlation heatmap
- Preprocessing: LabelEncoder + StandardScaler + SMOTE
- Models: LogisticRegression, RandomForest, XGBoost (5-Fold CV)
- Tuning: GridSearchCV Best Params: {'max_depth': 6, 'min_samples_split': 2, 'n_estimators': 100}

## Results
- Test Accuracy: 100.00%
- ROC-AUC: 1.000
- Key Drivers: Ki67, Grade, HER2, ER - matches clinical knowledge

## Deployment
model.pkl, scaler.pkl, columns.json, app.py (Streamlit)
Predicts with Confidence + Treatment: Luminal A->Hormone, HER2->Trastuzumab, TN->Chemo

## Conclusion
RF + Tuning achieved ~95%+ accuracy, ready for prototype.

## Future Scope
TCGA RNA-seq, CNN on histology, survival prediction.
