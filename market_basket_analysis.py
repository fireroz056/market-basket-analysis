import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load transaction data
transactions_df = pd.read_csv("transactions.csv")

# Load product metadata
product_metadata = pd.read_csv("product_metadata.csv")

# Convert transaction data into one-hot encoding (excluding Transaction_ID)
basket = transactions_df.drop(columns=['Transaction_ID']).astype(bool)

# Apply Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(basket, min_support=0.02, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Extract consequents (recommended items) as a single column
rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0] if len(x) == 1 else None)

# Merge with product metadata to factor in business metrics
rules = rules.merge(product_metadata, left_on='consequents', right_on='Item', how='left')

# Compute Weighted Score based on business impact (profitability, AOV, and conversion rate)
rules["Weighted_Score"] = (
    (rules["Profit_Margin"] * 0.5) + 
    (rules["Avg_Order_Value_Impact"] * 0.3) + 
    (rules["Conversion_Rate"] * 0.2)
)

# Sort rules by Weighted Score for best upsell recommendations
rules = rules.sort_values(by="Weighted_Score", ascending=False)

# Print the top recommendations instead of using ace_tools
print("Top Market Basket Recommendations:")
print(rules.head(10))  # Show top 10 recommendations

# Save results for reference
rules.to_csv("optimized_recommendations.csv", index=False)

