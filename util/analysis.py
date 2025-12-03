# analyze_infrastructure.py
import argparse
from infrastructure_analyzer import InfrastructureAnalyzer

def main():
    parser = argparse.ArgumentParser(description='Analyze infrastructure metrics for multi-agent systems')
    parser.add_argument('--metrics_dir', type=str, default='metrics',
                       help='Directory containing metrics JSON files')
    parser.add_argument('--methods', type=str, nargs='+',
                       help='Specific methods to analyze (e.g., latent_mas text_mas baseline)')
    parser.add_argument('--generate_report', action='store_true',
                       help='Generate detailed analysis report')
    
    args = parser.parse_args()
    
    analyzer = InfrastructureAnalyzer(metrics_dir=args.metrics_dir)
    
    methods_to_analyze = args.methods if args.methods else list(analyzer.data.keys())
    
    if not methods_to_analyze:
        print("No metrics data found. Please run experiments first.")
        return
    
    print(f"Analyzing methods: {methods_to_analyze}")
    
    # 生成对比图表
    analyzer.compare_methods(methods_to_analyze)
    
    # 生成详细报告
    if args.generate_report:
        analyzer.generate_detailed_report()
        print("Detailed report generated: infrastructure_analysis_report.json")
        analyzer.generate_energy_efficiency_report()
        print("Energy efficiency report generated: energy_efficiency_report.json")
        

if __name__ == "__main__":
    main()