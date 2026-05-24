# WMH-subtypes
This repository provides the analysis scripts, processing pipelines, and supporting results for the study “Lesion-level subtypes of white matter hyperintensity evolution beyond spatial location”.

The file WMH_clusters_metrics-all_with_clinic_final.xlsx contains all the data used in the analyses. This dataset includes multimodal metrics for each identified cluster, along with demographic and clinical information at the subject level.

Preprocessing

A folder named preprocessing is provided with scripts used to identify WMH clusters and extract multimodal signals.

Cluster identification
The script Whm_mask_only performs the initial cluster extraction from WMH lesions. MRI data used in this study can be requested through the Laboratory of Neuro Imaging (LONI) website. By using the subject IDs and acquisition dates provided in the excel (WMH_long_demografia+clinic), it is possible to obtain an exact replica of the MRI data used in the analyses.

This script:

Identifies lesion clusters.

Generates binary individual masks per subject containing the detected clusters.

Saves an initial Excel file containing cluster information.

Signal extraction by modality
The following scripts use the individual cluster masks generated in the previous step together with the Excel file to extract signals for each modality:

Whm_mask_only_mean_cluster_T1 – extracts the signal from T1-weighted images

Whm_mask_only_mean_cluster_DWI – extracts the signal from diffusion MRI (DWI)

Whm_mask_only_mean_cluster_fMRI – extracts the signal from functional MRI (fMRI)

Each script computes the mean signal for every cluster and stores the results for downstream analyses.

Cluster visualization
The script Whm_mask_only_plot_clusters generates visualizations of WMH lesions by subtype after clustering.

Input:

The Excel file located in the Results folder produced after clustering.

Output:

Plots illustrating lesion distribution across the different cluster-defined subtypes.

Analysis workflow

Cluster analysis
Run the script clustering using the Excel file as input.

This script performs:

The clustering analyses

The associated statistical tests

Susceptibility association analysis
After completing the clustering step, run the script susceptibility

This script uses the outputs generated in the previous step to perform the susceptibility association analyses.

Results

The folder Results_final contains all the results generated after running the clustering and susceptibility analyses.
