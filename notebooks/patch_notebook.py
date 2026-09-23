import nbformat

def update_notebook():
    nb_path = 'notebooks/Search Interest Forecasting.ipynb'
    nb = nbformat.read(nb_path, as_version=4)

    # 1. Fix Grouped ffill in Cell 24
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'ffill_cols =' in cell.source and 'ffill()' in cell.source:
            cell.source = '''ffill_cols = ["tone_net_sentiment","tone_positive_score","tone_negative_score","tone_polarity","tone_activity_density","tone_self_group_density"]\n# FIX: Perform ffill grouped by series to avoid cross-series data bleed\nfor col in ffill_cols:\n    df[col] = df.groupby(['Country_Name', 'Category'])[col].transform(lambda x: x.ffill().bfill())'''
            break

    # 2. Fix Drop contemporaneous features in Cell 46
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'df_model=df_model.drop(columns=unnecessary_cols)' in cell.source:
            cell.source = '''# Removing unnnecesay columns and PREVENTING DATA LEAKAGE
unnecessary_cols = ['tone_positive_score','tone_negative_score', 'tone_polarity', 
                    'tone_activity_density','tone_self_group_density', 'Source_Diversity',
                    'Media_Volume_per_Source', 'Interest_per_Source', 'Year', 
                    'Internet_Penetration','is_holiday']

# CRITICAL FIX: Drop contemporaneous features to prevent target leakage!
# We can't use these at time 't' to predict 't'
leakage_cols = ['Search_Velocity', 'Search_Acceleration', 'Media_Volume', 'tone_net_sentiment']

df_model = df_model.drop(columns=unnecessary_cols + leakage_cols)'''
            break

    # 3. Categorical encoding (Replacing Scaling cells)
    scaling_start = None
    scaling_end = None
    for i, cell in enumerate(nb.cells):
        if 'features_to_scale = ' in cell.source and 'excluded_cols =' in cell.source:
            scaling_start = i
        if 'scaler = StandardScaler()' in cell.source:
            scaling_end = i
            break
            
    if scaling_start is not None and scaling_end is not None:
        # Replace the scaling block with categorical encoding
        encoding_cell = nbformat.v4.new_code_cell(source='''# Encode Categorical Variables for LightGBM
# LightGBM handles categoricals well if converted to 'category' dtype
df_model['Country_Name'] = df_model['Country_Name'].astype('category')
df_model['Category'] = df_model['Category'].astype('category')
df_model['Country_Code'] = df_model['Country_Code'].astype('category')

print(df_model.dtypes)''')
        # Remove scaling cells and insert encoding cell
        del nb.cells[scaling_start:scaling_end+2] # Also remove cell 51
        nb.cells.insert(scaling_start, encoding_cell)

    # 4. Add Train/Val/Test Split and LightGBM Training
    idx_prep = None
    for i, cell in enumerate(nb.cells):
        if '3.1.4 X & y data preparation:' in cell.source:
            idx_prep = i
            break
            
    if idx_prep is not None:
        # Delete any empty cells after prep
        del nb.cells[idx_prep+1:]
        
        split_code = '''# Chronological Train / Val / Test Split
# Sort chronologically
df_model = df_model.sort_values('Week_Start').reset_index(drop=True)

# Define cutoffs
train_end = '2025-06-30'
val_end = '2025-12-31'

train_df = df_model[df_model['Week_Start'] <= train_end]
val_df = df_model[(df_model['Week_Start'] > train_end) & (df_model['Week_Start'] <= val_end)]
test_df = df_model[df_model['Week_Start'] > val_end]

print(f"Train size: {len(train_df)}")
print(f"Val size: {len(val_df)}")
print(f"Test size: {len(test_df)}")

target = 'Search_Interest'
features = [col for col in df_model.columns if col not in ['Week_Start', target]]

X_train, y_train = train_df[features], train_df[target]
X_val, y_val = val_df[features], val_df[target]
X_test, y_test = test_df[features], test_df[target]'''
        
        lgb_code = '''import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# LightGBM parameters
params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'verbose': -1
}

# Train the model
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)]
)'''
        
        eval_code = '''# Evaluate on Test Set
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Naive Baseline: Predict last week's value (Search_Interest_lag_1)
naive_pred = X_test['Search_Interest_lag_1']
naive_mae = mean_absolute_error(y_test, naive_pred)

print(f"--- Results ---")
print(f"LightGBM MAE: {mae:.2f}")
print(f"LightGBM RMSE: {rmse:.2f}")
print(f"Naive Baseline MAE: {naive_mae:.2f}")
print(f"Improvement over Baseline: {naive_mae - mae:.2f} points")'''

        nb.cells.extend([
            nbformat.v4.new_code_cell(source=split_code),
            nbformat.v4.new_markdown_cell(source='### 🚀 3.2 LightGBM Training'),
            nbformat.v4.new_code_cell(source=lgb_code),
            nbformat.v4.new_markdown_cell(source='### 📊 3.3 Evaluation & Comparison'),
            nbformat.v4.new_code_cell(source=eval_code)
        ])

    nbformat.write(nb, nb_path)
    print("Notebook updated successfully!")

if __name__ == '__main__':
    update_notebook()
