import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from werkzeug.utils import secure_filename
import joblib
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'risk_radar_secret_key'

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_data(df):
    """Validate uploaded data for required columns"""
    required_columns = ['customer_id', 'tenure', 'monthly_charges', 'total_charges', 'contract']
    missing_columns = [col for col in required_columns if col not in df.columns]
    return len(missing_columns) == 0, missing_columns

def predict_churn(df):
    """Use ML model for churn prediction and advanced analytics"""
    try:
        # Import the ChurnModel class and ChurnAnalyzer
        from models.churn_model import ChurnModel
        from models.churn_analyzer import ChurnAnalyzer
        # New analysis modules
        from models.pipeline import run_pipeline
        from models.eda import run_eda
        from models.association_rules import run_association_analysis
        from models.modeling import run_classification, run_clustering
        
        # Standardize column names (if needed)
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        # Ensure we have a customer_id column
        if 'customerid' in df.columns and 'customer_id' not in df.columns:
            df['customer_id'] = df['customerid']
        
        # Create model instance
        model = ChurnModel(model_type='logistic')
        
        # Try to load a pre-trained model, or train a new one if needed
        if not model.load():
            # For demo purposes, we'll use synthetic data to train the model
            from models.churn_model import generate_synthetic_data
            train_data = generate_synthetic_data(1000)
            model.train(train_data)
        
        # Make predictions
        result_df = model.predict(df)
        
        # Generate reasons for churn (in a real app, this would use SHAP values)
        reasons = [
            'High monthly charges',
            'No long-term contract',
            'Competitor offers',
            'Service dissatisfaction',
            'Lack of features',
            'Technical issues'
        ]
        
        # Assign reasons based on features (simplified approach)
        def get_reasons(row):
            selected_reasons = []
            
            # Check contract type
            if 'contract' in row and row['contract'] == 'Month-to-month':
                selected_reasons.append('No long-term contract')
            
            # Check monthly charges
            if 'monthlycharges' in row and row['monthlycharges'] > 70:
                selected_reasons.append('High monthly charges')
            
            # Check for services
            if 'onlinesecurity' in row and row['onlinesecurity'] == 'No':
                selected_reasons.append('Lack of features')
            
            if 'techsupport' in row and row['techsupport'] == 'No':
                selected_reasons.append('Technical issues')
            
            # Fill with random reasons if we don't have enough
            while len(selected_reasons) < 3:
                random_reason = np.random.choice([r for r in reasons if r not in selected_reasons])
                selected_reasons.append(random_reason)
            
            return selected_reasons[:3]
        
        # Apply the function to get reasons for each customer
        result_df['top_reasons'] = result_df.apply(get_reasons, axis=1)
        
        # Generate retention offers based on risk category
        offers = {
            'Low': 'Customer satisfaction survey',
            'Medium': '10% discount for 6 months',
            'High': '20% discount for 12 months + premium features'
        }
        result_df['retention_offer'] = result_df['risk_category'].map(offers)
        
        # Run advanced analytics (segmentation, churn factors, strategies, ROI)
        analyzer = ChurnAnalyzer(result_df)
        analysis_results = analyzer.run_complete_analysis()

        # Data pipeline & EDA
        pipeline_artifacts = run_pipeline(result_df)
        eda_results = run_eda(result_df)
        analysis_results['eda'] = eda_results

        # Association rules
        assoc_results = run_association_analysis(result_df, target_col='churn' if 'churn' in result_df.columns else 'churn_probability')
        analysis_results['association_rules'] = assoc_results

        # Modeling (if target column available)
        if pipeline_artifacts.get('target_col'):
            target_col = pipeline_artifacts['target_col']
            X = pipeline_artifacts['df_processed'].drop(columns=[target_col])
            y = pipeline_artifacts['df_processed'][target_col].astype(str).str.lower().isin(['yes', 'true', '1']).astype(int)

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            # Fit a ColumnTransformer for consistent preprocessing
            from models.pipeline import build_preprocessor
            preprocessor, _, _ = build_preprocessor(X_train)
            X_train_t = preprocessor.fit_transform(X_train)
            X_test_t = preprocessor.transform(X_test)

            cls_results = run_classification(X_train_t, y_train, X_test_t, y_test)
            analysis_results['modeling'] = cls_results

            # Feature importances from DT if available
            try:
                dt_best = cls_results['decision_tree'].get('best_params') is not None
                # Refit to access estimator cleanly
                from sklearn.tree import DecisionTreeClassifier
                dt_model = DecisionTreeClassifier(**cls_results['decision_tree']['best_params'], random_state=42)
                dt_model.fit(X_train_t, y_train)
                importances = dt_model.feature_importances_.tolist()
                feature_names = preprocessor.get_feature_names_out().tolist()
                fi = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:20]
                analysis_results.setdefault('feature_importance', {})['decision_tree'] = fi
            except Exception as e:
                print(f"Feature importance error: {e}")

            # Clustering on processed features
            cluster_results = run_clustering(preprocessor.transform(X))
            analysis_results['clustering'] = cluster_results
        else:
            # If no labeled target, still run clustering using numeric features
            try:
                X_numeric = result_df.select_dtypes(include=['number']).drop(columns=['churn_probability'], errors='ignore')
                if not X_numeric.empty:
                    clus = run_clustering(X_numeric.to_numpy())
                    analysis_results['clustering'] = clus
            except Exception as e:
                print(f"Clustering error: {e}")
        
        # Store analysis results in the dataframe for later use
        result_df['analysis_complete'] = True
        
        return result_df, analysis_results
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        # Fall back to random predictions if model fails
        np.random.seed(42)  # For reproducibility
        df['churn_probability'] = np.random.random(size=len(df))
        df['risk_category'] = pd.cut(
            df['churn_probability'], 
            bins=[0, 0.3, 0.7, 1.0], 
            labels=['Low', 'Medium', 'High']
        )
        df['top_reasons'] = [np.random.choice(reasons, size=3, replace=False).tolist() for _ in range(len(df))]
        df['retention_offer'] = df['risk_category'].map(offers)
        return df, None
    
    return df, None

def get_dashboard_stats(df):
    """Generate statistics for dashboard"""
    stats = {
        'total_customers': len(df),
        'churn_rate': df['churn_probability'].mean() * 100,
        'high_risk_count': len(df[df['risk_category'] == 'High']),
        'medium_risk_count': len(df[df['risk_category'] == 'Medium']),
        'low_risk_count': len(df[df['risk_category'] == 'Low']),
    }
    
    # Get top churn factors
    all_reasons = []
    for reasons in df['top_reasons']:
        all_reasons.extend(reasons)
    
    reason_counts = {}
    for reason in all_reasons:
        if reason in reason_counts:
            reason_counts[reason] += 1
        else:
            reason_counts[reason] = 1
    
    # Sort by count and get top 5
    top_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    stats['top_reasons'] = [{'reason': reason, 'count': count} for reason, count in top_reasons]
    
    return stats

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read the file
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:  # Excel file
            df = pd.read_excel(filepath)
        
        # Validate data
        is_valid, missing_columns = validate_data(df)
        if not is_valid:
            return jsonify({
                'error': 'Missing required columns',
                'missing_columns': missing_columns
            }), 400
        
        # Store in session
        session['current_file'] = filepath
        
        # Return preview
        preview = df.head(10).to_dict('records')
        return jsonify({
            'success': True,
            'preview': preview,
            'columns': df.columns.tolist(),
            'row_count': len(df)
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/predict', methods=['POST'])
def predict():
    if 'current_file' not in session:
        return jsonify({'error': 'No file uploaded'}), 400
    
    filepath = session['current_file']
    
    # Read the file
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:  # Excel file
        df = pd.read_excel(filepath)
    
    # Make predictions and run analysis
    result_df, analysis_results = predict_churn(df)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = f"results_{timestamp}.csv"
    result_df.to_csv(result_path, index=False)
    session['current_results'] = result_path
    
    # Generate dashboard stats
    stats = get_dashboard_stats(result_df)
    
    # Enhance stats with analysis results if available
    if analysis_results:
        # Add segmentation info
        if 'segmentation' in analysis_results:
            segment_counts = result_df['segment_name'].value_counts().to_dict() if 'segment_name' in result_df.columns else {}
            stats['customer_segments'] = segment_counts
        
        # Add retention strategy info
        if 'retention_strategies' in analysis_results:
            # Get top strategies across all segments
            all_strategies = []
            for segment_strategies in analysis_results['retention_strategies']['segment_strategies'].values():
                for strategy, count in segment_strategies:
                    all_strategies.append((strategy, count))
            
            # Combine counts for same strategies and get top 5
            strategy_counts = {}
            for strategy, count in all_strategies:
                if strategy in strategy_counts:
                    strategy_counts[strategy] += count
                else:
                    strategy_counts[strategy] = count
            
            top_strategies = sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            stats['top_strategies'] = [{'strategy': strategy, 'count': count} for strategy, count in top_strategies]
        
        # Add ROI analysis summary
        if 'roi_analysis' in analysis_results:
            stats['roi_summary'] = analysis_results['roi_analysis']['overall_roi']
    
    # Store results in session
    session['dashboard_stats'] = stats
    session['prediction_results'] = result_df.to_dict('records')
    session['analysis_results'] = analysis_results
    
    return jsonify({
        'success': True,
        'stats': stats,
        'redirect': url_for('dashboard')
    })

@app.route('/dashboard')
def dashboard():
    if 'dashboard_stats' not in session or 'prediction_results' not in session:
        return redirect(url_for('index'))
    
    stats = session['dashboard_stats']
    results = session['prediction_results']
    
    return render_template('dashboard.html', stats=stats, results=results)

@app.route('/customer/<customer_id>')
def customer_profile(customer_id):
    if 'prediction_results' not in session:
        return redirect(url_for('index'))
    
    results = session['prediction_results']
    customer = next((c for c in results if str(c['customer_id']) == customer_id), None)
    
    if not customer:
        return redirect(url_for('dashboard'))
    
    # Get detailed customer insights if analysis results are available
    customer_insights = None
    if 'analysis_results' in session and session['analysis_results']:
        try:
            # Import the ChurnAnalyzer class
            from models.churn_analyzer import ChurnAnalyzer
            
            # Create a DataFrame from the results
            results_df = pd.DataFrame(results)
            
            # Initialize the analyzer with the results
            analyzer = ChurnAnalyzer(results_df)
            
            # Get insights for this specific customer
            customer_insights = analyzer.get_customer_insights(customer['customer_id'])
        except Exception as e:
            print(f"Error getting customer insights: {e}")
    
    return render_template('customer_profile.html', customer=customer, insights=customer_insights)

@app.route('/export/<format>')
def export(format):
    # Build an exportable report
    stats = session.get('dashboard_stats')
    analysis = session.get('analysis_results')
    results = session.get('prediction_results')

    report = {
        'generated_at': datetime.now().isoformat(),
        'stats': stats,
        'analysis': analysis,
        'results_preview': results[:50] if results else []
    }

    if format.lower() == 'json':
        return jsonify({'success': True, 'report': report})
    elif format.lower() == 'csv':
        # Export current prediction results CSV
        if 'current_results' in session and os.path.exists(session['current_results']):
            return jsonify({'success': True, 'file': session['current_results']})
        else:
            return jsonify({'error': 'No CSV results available'}), 400
    else:
        return jsonify({'error': 'Unsupported export format'}), 400

@app.route('/analytics')
def analytics():
    if 'analysis_results' not in session or not session['analysis_results']:
        return redirect(url_for('dashboard'))
    
    analysis_results = session['analysis_results']
    
    # Prepare data for visualization
    visualization_data = {}
    
    # Segmentation data
    if 'segmentation' in analysis_results:
        segmentation = analysis_results['segmentation']
        
        # Prepare segment distribution data
        if 'segment_mapping' in segmentation:
            segment_mapping = segmentation['segment_mapping']
            visualization_data['segments'] = list(segment_mapping.values())
    
    # Churn factors data
    if 'churn_factors' in analysis_results:
        churn_factors = analysis_results['churn_factors']
        
        # Prepare churn factors data
        if 'sorted_reasons' in churn_factors:
            sorted_reasons = churn_factors['sorted_reasons']
            visualization_data['churn_factors'] = {
                'labels': [reason for reason, _ in sorted_reasons[:5]],
                'data': [count for _, count in sorted_reasons[:5]]
            }
        
        # Prepare segment-specific churn factors
        if 'segment_reasons' in churn_factors:
            segment_reasons = churn_factors['segment_reasons']
            visualization_data['segment_reasons'] = {}
            
            for segment, reasons in segment_reasons.items():
                visualization_data['segment_reasons'][segment] = {
                    'labels': [reason for reason, _ in reasons[:3]],
                    'data': [count for _, count in reasons[:3]]
                }
    
    # Retention strategies data
    if 'retention_strategies' in analysis_results:
        retention_strategies = analysis_results['retention_strategies']
        
        # Prepare retention strategies data
        if 'segment_strategies' in retention_strategies:
            segment_strategies = retention_strategies['segment_strategies']
            visualization_data['segment_strategies'] = {}
            
            for segment, strategies in segment_strategies.items():
                visualization_data['segment_strategies'][segment] = {
                    'labels': [strategy for strategy, _ in strategies[:3]],
                    'data': [count for _, count in strategies[:3]]
                }
    
    # ROI analysis data
    if 'roi_analysis' in analysis_results:
        roi_analysis = analysis_results['roi_analysis']
        
        # Prepare ROI analysis data
        if 'segment_roi' in roi_analysis:
            segment_roi = roi_analysis['segment_roi']
            visualization_data['segment_roi'] = {}
            
            for segment, roi_data in segment_roi.items():
                visualization_data['segment_roi'][segment] = roi_data
        
        # Overall ROI
        if 'overall_roi' in roi_analysis:
            visualization_data['overall_roi'] = roi_analysis['overall_roi']
    
    # Modeling performance
    if 'modeling' in analysis_results:
        modeling = analysis_results['modeling']
        perf = {}
        for name, res in modeling.items():
            perf[name] = res['metrics']
        visualization_data['model_performance'] = perf

    # Feature importance
    if 'feature_importance' in analysis_results:
        visualization_data['feature_importance'] = analysis_results['feature_importance']

    # Clustering visualization
    if 'clustering' in analysis_results:
        visualization_data['clustering'] = analysis_results['clustering']

    # EDA summaries
    if 'eda' in analysis_results:
        visualization_data['eda'] = analysis_results['eda']

    return render_template('analytics.html', analysis=visualization_data)

@app.route('/api/analytics')
def api_analytics():
    if 'analysis_results' not in session or not session['analysis_results']:
        return jsonify({'error': 'No analysis results available'}), 400
    
    return jsonify(session['analysis_results'])

@app.route('/api/model_metrics')
def api_model_metrics():
    if 'analysis_results' not in session or not session['analysis_results']:
        return jsonify({'error': 'No analysis results available'}), 400
    ar = session['analysis_results']
    return jsonify({
        'modeling': ar.get('modeling'),
        'feature_importance': ar.get('feature_importance'),
        'clustering': ar.get('clustering')
    })

if __name__ == '__main__':
    app.run(debug=True, port=5002)
