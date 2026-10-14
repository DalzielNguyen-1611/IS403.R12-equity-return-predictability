import os
import sys
import argparse
import logging

try:
    from . import test_config as config
    from . import test_engine as engine
    from . import test_analysis as analysis
except (ImportError, ValueError):
    import test_config as config
    import test_engine as engine
    import test_analysis as analysis


def setup_test_logger():
    logger = logging.getLogger("test_evaluator")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        log_file = os.path.join(config.TEST_RESULTS_DIR, 'test_execution.log')
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="Out-of-Sample Test Evaluation Runner (Split 3: 2020-2026)")
    parser.add_argument('--models', type=str, default='all',
                        help="Comma-separated models or 'all' (options: " + ", ".join(config.MODELS) + ")")
    parser.add_argument('--horizons', type=str, default='all',
                        help="Comma-separated horizons or 'all' (options: " + ", ".join(config.HORIZONS) + ")")
    parser.add_argument('--device', type=str, default=config.DEVICE,
                        help="Device to use for PyTorch models ('cpu' or 'cuda')")
    parser.add_argument('--report-only', action='store_true',
                        help="Skip model evaluation and regenerate analysis metrics and markdown report from existing predictions")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_test_logger()
    
    logger.info("=" * 70)
    logger.info("STARTING OUT-OF-SAMPLE TEST EVALUATION (SPLIT 3: 2020-2026)")
    logger.info("=" * 70)
    
    if args.report_only:
        logger.info("Mode: --report-only. Recomputing metrics and generating report...")
        if not os.path.exists(config.TEST_PREDICTIONS_PATH):
            logger.error(f"Cannot generate report: {config.TEST_PREDICTIONS_PATH} does not exist.")
            sys.exit(1)
        df_reg, df_cls, df_stat, df_regime = analysis.compute_metrics_and_tests()
        analysis.generate_markdown_report(df_reg, df_cls, df_stat, df_regime)
        logger.info(f"Report generated successfully at: {config.TEST_REPORT_PATH}")
        return
        
    models = config.MODELS if args.models.lower() == 'all' else [m.strip().lower() for m in args.models.split(',')]
    horizons = config.HORIZONS if args.horizons.lower() == 'all' else [config.normalize_horizon(h) for h in args.horizons.split(',')]
    
    # 1. Run model evaluations
    preds_df = engine.run_test_evaluation(models=models, horizons=horizons, logger=logger, device=args.device)
    
    # 2. Compute metrics, statistical tests, and regime breakdown
    logger.info("\nComputing statistical inference tests (Diebold-Mariano & Sign Tests) and regime breakdowns...")
    df_reg, df_cls, df_stat, df_regime = analysis.compute_metrics_and_tests(preds_df)
    
    # 3. Generate Markdown Report
    report_content = analysis.generate_markdown_report(df_reg, df_cls, df_stat, df_regime)
    logger.info(f"\nSaved Markdown Report to: {config.TEST_REPORT_PATH}")
    
    logger.info("=" * 70)
    logger.info("OUT-OF-SAMPLE TEST EVALUATION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
