import json
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import warnings
import glob
from collections import defaultdict
import gc

warnings.filterwarnings("ignore")


class DOVEBatchProcessor:
    """
    Process DOVE parquet files in batches to avoid memory issues
    Uses intermediate files to accumulate scores
    """

    def __init__(self, data_directory="olmoe_mmlu_data", batch_size=4):
        self.data_directory = Path(data_directory)
        self.batch_size = batch_size
        self.model_name = "OLMoE-1B-7B-0924-Instruct"
        self.shots = [0, 5]

        # Intermediate file to store accumulated scores
        self.intermediate_file = "score_accumulator.json"

        print(f"Initialized batch processor")
        print(f"Data directory: {self.data_directory}")
        print(f"Batch size: {batch_size} files")
        print(f"Intermediate file: {self.intermediate_file}")

    def discover_parquet_files(self):
        """
        Discover all parquet files and organize them by shot
        """
        print("🔍 Discovering parquet files...")

        if not self.data_directory.exists():
            print(f"❌ Data directory does not exist: {self.data_directory}")
            return []

        all_files = []

        # Pattern: {shots}_shot_mmlu.{subject}.parquet
        for shot in self.shots:
            pattern = f"{shot}_shot_mmlu.*.parquet"
            file_pattern = self.data_directory / pattern
            files = glob.glob(str(file_pattern))

            for file_path in files:
                filename = Path(file_path).name
                subject = filename.replace(f"{shot}_shot_", "").replace(".parquet", "")

                all_files.append({
                    'file_path': file_path,
                    'shot': shot,
                    'subject': subject,
                    'filename': filename
                })

        print(f"✅ Found {len(all_files)} total files")

        # Show breakdown by shot
        shot_counts = {}
        for file_info in all_files:
            shot = file_info['shot']
            shot_counts[shot] = shot_counts.get(shot, 0) + 1

        for shot, count in shot_counts.items():
            print(f"  {shot}-shot: {count} files")

        return all_files

    def create_batches(self, all_files):
        """
        Split files into batches
        """
        batches = []
        for i in range(0, len(all_files), self.batch_size):
            batch = all_files[i:i + self.batch_size]
            batches.append(batch)

        print(f"📦 Created {len(batches)} batches of up to {self.batch_size} files each")
        return batches

    def load_or_create_accumulator(self):
        """
        Load existing score accumulator or create new one
        Format: {sample_index: {"sum": total_score, "count": num_instances}}
        """
        if os.path.exists(self.intermediate_file):
            print(f"📂 Loading existing accumulator: {self.intermediate_file}")
            with open(self.intermediate_file, 'r') as f:
                accumulator = json.load(f)

            # Convert to defaultdict for easier handling
            score_accumulator = defaultdict(lambda: {"sum": 0.0, "count": 0})
            for sample_idx, data in accumulator.items():
                score_accumulator[sample_idx] = data

            total_samples = len(score_accumulator)
            total_instances = sum(data["count"] for data in score_accumulator.values())
            print(f"  Loaded {total_samples} samples with {total_instances} total instances")

        else:
            print(f"🆕 Creating new accumulator")
            score_accumulator = defaultdict(lambda: {"sum": 0.0, "count": 0})

        return score_accumulator

    def save_accumulator(self, score_accumulator):
        """
        Save current state of score accumulator
        """
        # Convert defaultdict to regular dict for JSON serialization
        regular_dict = dict(score_accumulator)

        with open(self.intermediate_file, 'w') as f:
            json.dump(regular_dict, f, indent=2)

        total_samples = len(regular_dict)
        total_instances = sum(data["count"] for data in regular_dict.values())
        print(f"💾 Saved accumulator: {total_samples} samples, {total_instances} instances")

    def process_batch(self, batch, batch_num, total_batches):
        """
        Process a single batch of files
        """
        print(f"\n📦 Processing batch {batch_num}/{total_batches}")
        print(f"Files in this batch:")
        for file_info in batch:
            print(f"  - {file_info['filename']}")

        batch_data = []

        # Load all files in the batch
        for file_info in tqdm(batch, desc=f"Loading batch {batch_num}"):
            try:
                # Load parquet file
                df = pd.read_parquet(file_info['file_path'])

                # Add metadata
                df['shot'] = file_info['shot']
                df['subject'] = file_info['subject']

                batch_data.append(df)

            except Exception as e:
                print(f"❌ Error loading {file_info['file_path']}: {e}")
                continue

        if not batch_data:
            print(f"❌ No files loaded successfully in batch {batch_num}")
            return None

        # Combine batch data
        combined_df = pd.concat(batch_data, ignore_index=True)
        print(f"  Combined batch shape: {combined_df.shape}")

        # Clear batch_data to free memory
        del batch_data
        gc.collect()

        return combined_df

    def extract_scores_from_batch(self, df):
        """
        Extract scores from batch DataFrame and return accumulator updates
        """
        print("🔄 Extracting scores from batch...")

        # Find score column
        score_cols = ['score', 'average_score', 'mean_score']
        score_col = None

        for col in score_cols:
            if col in df.columns:
                score_col = col
                break

        if score_col is None:
            print(f"❌ No score column found. Available columns: {list(df.columns)}")
            score_like_cols = [col for col in df.columns if 'score' in col.lower()]
            if score_like_cols:
                score_col = score_like_cols[0]
                print(f"Using column: {score_col}")
            else:
                return None

        # Check for sample_index
        if 'sample_index' not in df.columns:
            print(f"❌ sample_index column not found. Available columns: {list(df.columns)}")
            return None

        # Clean data
        df_clean = df.dropna(subset=[score_col, 'sample_index'])
        df_clean['sample_index'] = df_clean['sample_index'].astype(str)

        print(f"  Processing {len(df_clean)} valid records")
        print(f"  Unique samples in batch: {df_clean['sample_index'].nunique()}")

        # Calculate batch accumulator
        batch_accumulator = {}

        for sample_idx, group in df_clean.groupby('sample_index'):
            scores = group[score_col].values
            batch_accumulator[sample_idx] = {
                "sum": float(scores.sum()),
                "count": len(scores)
            }

        print(f"  Extracted scores for {len(batch_accumulator)} samples")

        # Clear DataFrame to free memory
        del df, df_clean
        gc.collect()

        return batch_accumulator

    def update_main_accumulator(self, main_accumulator, batch_accumulator):
        """
        Update main accumulator with batch results
        """
        print("🔄 Updating main accumulator...")

        for sample_idx, batch_data in batch_accumulator.items():
            main_accumulator[sample_idx]["sum"] += batch_data["sum"]
            main_accumulator[sample_idx]["count"] += batch_data["count"]

        total_samples = len(main_accumulator)
        total_instances = sum(data["count"] for data in main_accumulator.values())
        print(f"  Main accumulator now has {total_samples} samples, {total_instances} total instances")

    def process_all_batches(self):
        """
        Process all batches and maintain accumulator
        """
        print("🚀 Starting batch processing...")

        # Discover files and create batches
        all_files = self.discover_parquet_files()
        if not all_files:
            return None

        batches = self.create_batches(all_files)

        # Load or create accumulator
        main_accumulator = self.load_or_create_accumulator()

        # Process each batch
        for batch_num, batch in enumerate(batches, 1):
            print(f"\n{'=' * 50}")
            print(f"BATCH {batch_num}/{len(batches)}")
            print('=' * 50)

            # Process batch
            batch_df = self.process_batch(batch, batch_num, len(batches))
            if batch_df is None:
                continue

            # Extract scores from batch
            batch_accumulator = self.extract_scores_from_batch(batch_df)
            if batch_accumulator is None:
                continue

            # Update main accumulator
            self.update_main_accumulator(main_accumulator, batch_accumulator)

            # Save accumulator after each batch (checkpoint)
            self.save_accumulator(main_accumulator)

            print(f"✅ Batch {batch_num} completed and saved")

        return main_accumulator

    def calculate_final_scores(self, accumulator):
        """
        Calculate final mean scores from accumulator
        """
        print("🔄 Calculating final mean scores...")

        final_scores = {}

        for sample_idx, data in accumulator.items():
            if data["count"] > 0:
                mean_score = data["sum"] / data["count"]
                final_scores[sample_idx] = round(mean_score, 4)

        print(f"✅ Calculated final scores for {len(final_scores)} samples")

        # Sort by sample_index (numerically if possible)
        try:
            sorted_items = sorted(final_scores.items(), key=lambda x: int(x[0]))
            final_scores = dict(sorted_items)
        except ValueError:
            final_scores = dict(sorted(final_scores.items()))

        return final_scores

    def save_final_json(self, final_scores, output_file="olmoe_mmlu_final_scores.json"):
        """
        Save final JSON output
        """
        print(f"💾 Saving final JSON: {output_file}")

        with open(output_file, 'w') as f:
            json.dump(final_scores, f, indent=2)

        print(f"✅ Final JSON saved with {len(final_scores)} entries")

        # Show first few examples
        print(f"\nFirst 10 entries:")
        for i, (sample_idx, score) in enumerate(list(final_scores.items())[:10]):
            print(f'  "{sample_idx}": {score}')

        return final_scores

    def create_summary_report(self, accumulator, final_scores):
        """
        Create a summary report
        """
        print("📊 Creating summary report...")

        total_samples = len(final_scores)
        total_instances = sum(data["count"] for data in accumulator.values())
        avg_instances_per_sample = total_instances / total_samples if total_samples > 0 else 0

        min_score = min(final_scores.values()) if final_scores else 0
        max_score = max(final_scores.values()) if final_scores else 0
        avg_score = sum(final_scores.values()) / len(final_scores) if final_scores else 0

        summary = {
            "processing_summary": {
                "total_samples": total_samples,
                "total_instances": total_instances,
                "avg_instances_per_sample": round(avg_instances_per_sample, 2)
            },
            "score_statistics": {
                "min_score": min_score,
                "max_score": max_score,
                "average_score": round(avg_score, 4)
            }
        }

        with open("olmoe_processing_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print("✅ Summary report saved: olmoe_processing_summary.json")
        print(f"  Total samples: {total_samples}")
        print(f"  Total instances: {total_instances}")
        print(f"  Average score: {avg_score:.4f}")

    def run_complete_pipeline(self):
        """
        Run the complete batch processing pipeline
        """
        print("🚀 Starting DOVE Batch Processing Pipeline")
        print("=" * 60)

        try:
            # Process all batches
            accumulator = self.process_all_batches()
            if accumulator is None:
                print("❌ Batch processing failed")
                return None

            # Calculate final scores
            final_scores = self.calculate_final_scores(accumulator)

            # Save final JSON
            final_json = self.save_final_json(final_scores)

            # Create summary report
            self.create_summary_report(accumulator, final_scores)

            # Clean up intermediate file
            if os.path.exists(self.intermediate_file):
                os.remove(self.intermediate_file)
                print(f"🧹 Cleaned up intermediate file: {self.intermediate_file}")

            print("\n🎉 Pipeline completed successfully!")
            print("=" * 60)
            print(f"Final output: olmoe_mmlu_final_scores.json")
            print(f"Contains {len(final_scores)} sample scores")

            return final_scores

        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            print("💾 Intermediate results saved in score_accumulator.json")
            return None


def main():
    """
    Main function with options for batch size and resuming
    """
    import argparse

    parser = argparse.ArgumentParser(description='Process DOVE parquet files in batches')
    parser.add_argument('--batch-size', type=int, default=4, help='Number of files per batch (default: 4)')
    parser.add_argument('--data-dir', type=str, default='olmoe_mmlu_data',
                        help='Data directory (default: olmoe_mmlu_data)')
    parser.add_argument('--resume', action='store_true', help='Resume from existing intermediate file')

    args = parser.parse_args()

    print(f"Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Resume mode: {args.resume}")

    if not args.resume and os.path.exists("score_accumulator.json"):
        response = input("\nIntermediate file exists. Resume processing? (y/n): ")
        if response.lower() != 'y':
            os.remove("score_accumulator.json")
            print("Deleted existing intermediate file. Starting fresh.")

    # Initialize and run processor
    processor = DOVEBatchProcessor(
        data_directory=args.data_dir,
        batch_size=args.batch_size
    )

    result = processor.run_complete_pipeline()

    if result:
        print(f"\n✅ Successfully processed {len(result)} samples")
    else:
        print("\n❌ Processing failed - check intermediate files")


if __name__ == "__main__":
    main()
