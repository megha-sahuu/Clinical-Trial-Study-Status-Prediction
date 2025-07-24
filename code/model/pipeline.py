from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

def build_pipeline():
    """
    Build the preprocessing pipeline and classifier.
    """
    numeric_transformer = 'passthrough'
    text_transformer = TfidfVectorizer(max_features=500, stop_words='english')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, ['Enrollment']),
            ('text', text_transformer, 'combined_text')
        ]
    )
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', xgb)
    ])
    return pipeline
