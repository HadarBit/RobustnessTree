import json
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class DoveLiteMMPipelineLocal:
    """
    Complete pipeline to process local DOVE_Lite MMLU data for OLMoE-1B-7B-0924-Instruct
    """

    def __init__(self, base_path="app/results_local/nlphuji_DOVE_Lite/en"):
        self.base_path = Path(base_path)
        self.model_name = "OLMoE-1B-7B-0924-Instruct"
        self.language = "en"
        self.shots = [0, 5]

        # All 57 MMLU datasets (based on your directory structure)
        self.mmlu_datasets = [
            "mmlu.abstract_algebra",
            "mmlu.anatomy",
            "mmlu.astronomy",
            "mmlu.business_ethics",
            "mmlu.clinical_knowledge",
            "mmlu.college_biology",
            "mmlu.college_chemistry",
            "mmlu.college_computer_science",
            "mmlu.college_mathematics",
            "mmlu.college_medicine",
            "mmlu.college_physics",
            "mmlu.computer_security",
            "mmlu.conceptual_physics",
            "mmlu.econometrics",
            "mmlu.electrical_engineering",
            "mmlu.elementary_mathematics",
            "mmlu.formal_logic",
            "mmlu.global_facts",
            "mmlu.high_school_biology",
            "mmlu.high_school_chemistry",
            "mmlu.high_school_computer_science",
            "mmlu.high_school_european_history",
            "mmlu.high_school_geography",
            "mmlu.high_school_government_and_politics",
            "mmlu.high_school_macroeconomics",
            "mmlu.high_school_mathematics",
            "mmlu.high_school_microeconomics",
            "mmlu.high_school_physics",
            "mmlu.high_school_psychology",
            "mmlu.high_school_statistics",
            "mmlu.high_school_us_history",
            "mmlu.high_school_world_history",
            "mmlu.human_aging",
            "mmlu.human_sexuality",
            "mmlu.international_law",
            "mmlu.jurisprudence",
            "mmlu.logical_fallacies",
            "mmlu.machine_learning",
            "mmlu.management",
            "mmlu.marketing",
            "mmlu.medical_genetics",
            "mmlu.miscellaneous",
            "mmlu.moral_disputes",
            "mmlu.moral_scenarios",
            "mmlu.nutrition",
            "mmlu.philosophy",
            "mmlu.prehistory",
            "mmlu.professional_accounting",
            "mmlu.professional_law",
            "mmlu.professional_medicine",
            "mmlu.professional_psychology",
            "mmlu.public_relations",
            "mmlu.security_studies",
            "mmlu.sociology",
            "mmlu.us_foreign_policy",
            "mmlu.virology",
            "mmlu.world_religions",
        ]

        print(f"Initialized pipeline for {len(self.mmlu_datasets)} MMLU datasets")
        print(f"Model: {self.model_name}")
        print(f"Shots: {self.shots}")
        print(f"Base path: {self.base_path}")

    def load_local_data(self):
        """
        Load all DOVE_Lite data from local parquet files
        """
        print("🔄 Loading DOVE_Lite dataset from local files...")

        all_data = []

        # Check if base path exists
        if not self.base_path.exists():
            print(f"❌ Base path does not exist: {self.base_path}")
            return None

        # Process each shot configuration
        for shot in self.shots:
            shot_dir = self.base_path / f"Shots_{shot}" / self.model_name

            if not shot_dir.exists():
                print(f"⚠️  Shot directory does not exist: {shot_dir}")
                continue

            print(f"📁 Processing {shot}-shot data from: {shot_dir}")

            # Process each MMLU dataset
            for dataset in tqdm(self.mmlu_datasets, desc=f"Loading {shot}-shot data"):
                dataset_dir = shot_dir / dataset
                parquet_file = dataset_dir / "low_performance_questions_0.1.parquet"

                if parquet_file.exists():
                    try:
                        # Load the parquet file
                        df = pd.read_parquet(parquet_file)

                        # Add metadata columns to match HF format
                        df["model"] = self.model_name
                        df["dataset"] = dataset
                        df["dimensions_5: shots"] = shot
                        df["language"] = self.language

                        all_data.append(df)

                    except Exception as e:
                        print(f"❌ Error loading {parquet_file}: {e}")
                else:
                    print(f"⚠️  File not found: {parquet_file}")

        if not all_data:
            print("❌ No data files were loaded successfully!")
            return None

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        print(f"✅ Loaded {len(combined_df)} total records from local files")
        print(f"Available datasets: {combined_df['dataset'].nunique()} unique datasets")
        print(
            f"Shot breakdown: {combined_df['dimensions_5: shots'].value_counts().to_dict()}"
        )

        return combined_df

    def validate_data_structure(self, df):
        """
        Validate that the loaded data has the expected structure
        """
        print("🔄 Validating data structure...")

        required_columns = ["sample_index"]
        score_columns = ["score", "average_score"]  # Check for either column name

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return False

        # Check for score column (either 'score' or 'average_score')
        score_col_found = any(col in df.columns for col in score_columns)
        if not score_col_found:
            print(f"❌ Missing score column. Expected one of: {score_columns}")
            print(f"Available columns: {list(df.columns)}")
            return False

        # Rename 'average_score' to 'score' if needed for consistency
        if "average_score" in df.columns and "score" not in df.columns:
            df["score"] = df["average_score"]
            print("✅ Renamed 'average_score' to 'score' for consistency")

        print(f"✅ Data structure validated")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types: {df.dtypes.to_dict()}")

        return True

    def filter_data(self, df):
        """
        Filter data for our specific requirements (already filtered during loading)
        """
        print("🔄 Data already filtered during loading, validating...")

        print(f"Total records: {len(df)}")
        print(f"Model: {df['model'].unique()}")
        print(f"Datasets: {df['dataset'].nunique()} unique datasets")
        print(f"Shots: {sorted(df['dimensions_5: shots'].unique())}")

        # Show breakdown by dataset
        print(f"\nDataset breakdown:")
        dataset_counts = df["dataset"].value_counts()
        print(f"Found data for {len(dataset_counts)} MMLU datasets:")
        for dataset, count in dataset_counts.head(10).items():
            print(f"  {dataset}: {count} records")
        if len(dataset_counts) > 10:
            print(f"  ... and {len(dataset_counts) - 10} more datasets")

        # Show breakdown by shots
        print(f"\nShots breakdown:")
        shots_counts = df["dimensions_5: shots"].value_counts()
        for shots, count in shots_counts.items():
            print(f"  {shots}-shot: {count} records")

        return df

    def calculate_mean_scores(self, df):
        """
        Calculate mean scores for each sample_index across all permutations
        """
        print("🔄 Calculating mean scores per sample_index...")

        # Group by sample_index and calculate mean score
        # This averages across all dimensions: shots, datasets, and any other variations
        mean_scores = (
            df.groupby("sample_index")["score"]
            .agg(["mean", "count", "std"])
            .reset_index()
        )
        mean_scores.columns = ["sample_index", "mean_score", "count", "std_score"]

        print(f"✅ Calculated mean scores for {len(mean_scores)} unique sample indices")
        print(f"Score statistics:")
        print(f"  Mean score: {mean_scores['mean_score'].mean():.3f}")
        print(
            f"  Score range: {mean_scores['mean_score'].min():.3f} to {mean_scores['mean_score'].max():.3f}"
        )
        print(f"  Average records per sample: {mean_scores['count'].mean():.1f}")
        print(f"  Min records per sample: {mean_scores['count'].min()}")
        print(f"  Max records per sample: {mean_scores['count'].max()}")

        return mean_scores

    def create_json_output(
        self, mean_scores_df, output_file="mmlu_dove_lite_scores_local.json"
    ):
        """
        Create final JSON output with sample_index -> mean_score mapping
        """
        print(f"🔄 Creating JSON output...")

        # Create dictionary mapping sample_index to mean_score
        scores_dict = dict(
            zip(mean_scores_df["sample_index"], mean_scores_df["mean_score"])
        )

        # Save as JSON
        with open(output_file, "w") as f:
            json.dump(scores_dict, f, indent=2)

        print(f"✅ JSON file saved: {output_file}")
        print(f"   Contains {len(scores_dict)} sample indices")

        return scores_dict

    def save_detailed_results(self, df, mean_scores_df):
        """
        Save additional detailed files for analysis
        """
        print("🔄 Saving detailed results...")

        # Save full filtered data
        df.to_parquet("mmlu_dove_lite_full_data_local.parquet", index=False)
        print("✅ Saved full data: mmlu_dove_lite_full_data_local.parquet")

        # Save mean scores with statistics
        mean_scores_df.to_parquet(
            "mmlu_dove_lite_mean_scores_local.parquet", index=False
        )
        print("✅ Saved mean scores: mmlu_dove_lite_mean_scores_local.parquet")

        # Create summary by dataset
        dataset_summary = (
            df.groupby("dataset")
            .agg({"score": ["mean", "std", "count"], "sample_index": "nunique"})
            .round(3)
        )

        dataset_summary.columns = [
            "mean_score",
            "std_score",
            "total_records",
            "unique_samples",
        ]
        dataset_summary = dataset_summary.reset_index()

        # Save dataset summary
        dataset_summary.to_json(
            "mmlu_dove_lite_dataset_summary_local.json", orient="records", indent=2
        )
        print("✅ Saved dataset summary: mmlu_dove_lite_dataset_summary_local.json")

        # Create summary by shots
        shots_summary = (
            df.groupby("dimensions_5: shots")
            .agg({"score": ["mean", "std", "count"], "sample_index": "nunique"})
            .round(3)
        )

        shots_summary.columns = [
            "mean_score",
            "std_score",
            "total_records",
            "unique_samples",
        ]
        shots_summary = shots_summary.reset_index()

        print(f"\nShots performance summary:")
        for _, row in shots_summary.iterrows():
            print(
                f"  {int(row['dimensions_5: shots'])}-shot: {row['mean_score']:.3f} ± {row['std_score']:.3f} "
                f"({row['total_records']} records, {row['unique_samples']} unique samples)"
            )

    def run_complete_pipeline(self):
        """
        Run the complete pipeline
        """
        print("🚀 Starting DOVE_Lite MMLU Pipeline (Local Files)")
        print("=" * 60)

        # Step 1: Load data from local files
        df = self.load_local_data()
        if df is None:
            return None

        # Step 2: Validate data structure
        if not self.validate_data_structure(df):
            return None

        # Step 3: Filter/validate data (already filtered during loading)
        df_filtered = self.filter_data(df)
        if len(df_filtered) == 0:
            print("❌ No data found after filtering!")
            return None

        # Step 4: Calculate mean scores per sample_index
        mean_scores_df = self.calculate_mean_scores(df_filtered)

        # Step 5: Create final JSON output
        scores_dict = self.create_json_output(mean_scores_df)

        # Step 6: Save detailed results
        self.save_detailed_results(df_filtered, mean_scores_df)

        print("\n🎉 Pipeline completed successfully!")
        print("=" * 60)
        print(f"Final output: mmlu_dove_lite_scores_local.json")
        print(f"Contains {len(scores_dict)} sample indices with mean scores")

        # Show a few examples
        print(f"\nExample entries:")
        sample_items = list(scores_dict.items())[:5]
        for idx, score in sample_items:
            print(f"  Sample {idx}: {score:.3f}")

        return scores_dict


def main():
    """
    Main function to run the pipeline
    """
    # You can customize the base path here if needed
    base_path = "app/results_local/nlphuji_DOVE_Lite/en"

    # Initialize and run pipeline
    pipeline = DoveLiteMMPipelineLocal(base_path=base_path)
    result = pipeline.run_complete_pipeline()

    if result:
        print(f"\n✅ Successfully processed {len(result)} samples")
    else:
        print("\n❌ Pipeline failed")


if __name__ == "__main__":
    main()
