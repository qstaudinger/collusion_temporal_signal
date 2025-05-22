This folder contains the followings files which are described

DB_Collusion_Switzerland_GR_and_See-Gaster_processed.csv
Collusive dataset from Switzerland GR and See-Gaster. It has the columns:
Bid_value (Swiss Franc), Collusive_competitor, Contract_type, Tender, Date, Number_bids, Collusive_competitor_original, Winner and the screening variables (CV, SPD, DIFFP, RD, KURT, SKEW, KSTEST)

01_Data_Pipeline.ipynb
The notebook 01_Data_Pipeline.ipynb sets up the initial data pipeline for a project focused on detecting collusion in public procurement. It includes the import of essential libraries, the configuration of file paths, and the loading of a cleaned dataset containing bidding information, contract types, and indicators of potential collusive behavior. Using tools like ydata_profiling, the notebook provides an initial exploratory data analysis to assess data quality and structure. The resulting dataset forms the basis for subsequent modeling and machine learning applications aimed at identifying collusive patterns.

02_models.ipynb
The notebook 02_models.ipynb implements a machine learning workflow for detecting collusive behavior in procurement data. It sets up multiple classification models, including decision trees, random forests, gradient boosting, neural networks, and logistic regression. The models are trained using a unified pipeline structure with hyperparameter tuning via grid search and cross-validation. Performance is evaluated using metrics such as confusion matrices and ROC curves. This step builds on the cleaned dataset and prepares the ground for robust collusion detection through predictive modeling.


Notes:

 - Date is timestamp format
 - Winner is 1 (the bid is the winner of the tender) or 0 (the bid is not the winner of the tender)
 - Competitors is the ID for each company of the dataset
 - Collusive_competitor is 1 (collusive) or 0 (not collusive)
 - Collusive_competitor_original is the original collusion without our treshold (see paper): 1 collusive, 0 not collusive
 - Exchange rates to Euros applied (early 2021): 1 Swiss Franc (0.92€) and 1 US Dollar (0.0078€) 



Code based on: Wallimann, Hannes et al. (2025). “Where is the Limit? Assessing the Potential of Algorithm-Based Cartel Detection”. In: Journal of Competition Law & Economics. DOI: 10 . 1093 / joclec /
nhae023.

Data and Notes based on: García Rodríguez, Manuel J et al. (2022). “Collusion detection in public procurement auctions with machine learning algorithms”. In: Automation in Construction 133, p. 104047. DOI: 10.1016/j.autcon.2021.104047.

