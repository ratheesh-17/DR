Early Detection of Diabetic Retinopathy using CNN + Explainable AI
HealthcareComputer VisionCNNGrad-CAMXAI
What it is
A deep learning system that classifies retinal fundus images into 5 severity levels of diabetic retinopathy. It also generates visual heatmaps (Grad-CAM) showing which parts of the eye the model focused on — making predictions explainable to doctors.
Gap in existing research
Most existing models are black-boxes. Clinicians cannot trust a model they can't interpret. Also, class imbalance (very few severe cases) is rarely addressed properly in prior work.
How your project fills the gap
You add Explainable AI (Grad-CAM / LIME) + handle class imbalance with SMOTE or weighted loss. This makes it clinically deployable — a major research contribution angle from journals like IEEE Access / Nature Scientific Reports.
Dataset
APTOS 2019 Blindness Detection (Kaggle) — 3,662 labelled retinal images. Also IDRiD and EyePACS available publicly.
------>this was the project, we were going to do for final year

Technical Stacks are
Backend - FastAPI
Frontend - Only React not Vite
Local MySQL - username : root, password : Ratheesh@1703
