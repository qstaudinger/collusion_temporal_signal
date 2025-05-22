This folder contains the followings files which are described

Data: DB_Collusion_Switzerland_GR_and_See-Gaster_processed.csv
Collusive dataset from Switzerland GR and See-Gaster. It has the columns:
Bid_value (Swiss Franc), Collusive_competitor, Contract_type, Tender, Date, Number_bids, Collusive_competitor_original, Winner and the screening variables (CV, SPD, DIFFP, RD, KURT, SKEW, KSTEST)

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

