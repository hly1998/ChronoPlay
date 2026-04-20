#!/usr/bin/env python3
"""
Aggregate submission files from submissions/ directory to leaderboard.json

This script:
1. Reads all submissions/*.json files
2. Validates each submission format
3. Calculates average scores across all games
4. Merges with existing leaderboard.json (deduplication and updates)
5. Writes results back to leaderboard.json
"""

import json
from pathlib import Path
from typing import List, Dict, Any


REQUIRED_GAMES = ['dune', 'dying_light_2', 'pubg_mobile']
METRIC_FIELDS = ['topk', 'recall', 'f1', 'ndcg', 'correctness', 'faithfulness']


def validate_submission(data: Dict[str, Any], filename: str) -> bool:
    """Validate submission data format"""

    # Check required top-level fields
    required_fields = ['system_name', 'description', 'games']
    for field in required_fields:
        if field not in data:
            print(f"❌ Error: {filename} missing required field '{field}'")
            return False

    # Check games field
    if not isinstance(data['games'], dict):
        print(f"❌ Error: {filename} 'games' field must be a dictionary")
        return False

    # Check all required games are present
    for game in REQUIRED_GAMES:
        if game not in data['games']:
            print(f"❌ Error: {filename} missing game '{game}' in 'games' field")
            return False

    # Validate each game's metrics
    for game, metrics in data['games'].items():
        if not isinstance(metrics, dict):
            print(f"❌ Error: {filename} game '{game}' metrics must be a dictionary")
            return False

        # Check all required metric fields
        for field in METRIC_FIELDS:
            if field not in metrics:
                print(f"❌ Error: {filename} game '{game}' missing metric '{field}'")
                return False

            # Validate metric values
            try:
                value = float(metrics[field])

                # Check range for different metrics
                if field == 'topk':
                    # topk should be a positive integer, but we accept float and convert
                    if value <= 0:
                        print(f"❌ Error: {filename} game '{game}' '{field}' must be positive, got {value}")
                        return False
                else:
                    # Other metrics should be between 0-1
                    if not (0 <= value <= 1):
                        print(f"❌ Error: {filename} game '{game}' '{field}' must be between 0-1, got {value}")
                        return False
            except (ValueError, TypeError):
                print(f"❌ Error: {filename} game '{game}' '{field}' must be a number")
                return False

    return True


def calculate_average(games_data: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Calculate average metrics across all games"""
    average = {}

    for metric in METRIC_FIELDS:
        values = [games_data[game][metric] for game in REQUIRED_GAMES]
        if metric == 'topk':
            # For topk, we take the most common value, or the first one if all different
            average[metric] = int(values[0])
        else:
            average[metric] = sum(values) / len(values)

    return average


def load_submissions(submissions_dir: Path) -> List[Dict[str, Any]]:
    """Load all submission files"""
    submissions = []

    if not submissions_dir.exists():
        print(f"⚠️  Warning: {submissions_dir} directory does not exist")
        return submissions

    json_files = list(submissions_dir.glob('*.json'))

    # Exclude example file
    json_files = [f for f in json_files if f.name != 'example_submission.json']

    if not json_files:
        print("ℹ️  No new submission files found")
        return submissions

    print(f"📁 Found {len(json_files)} submission file(s)")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if validate_submission(data, json_file.name):
                # Calculate average if not provided
                if 'average' not in data:
                    data['average'] = calculate_average(data['games'])
                    print(f"📊 Calculated average scores for {json_file.name}")

                submissions.append(data)
                print(f"✅ Successfully loaded: {json_file.name}")
            else:
                print(f"⚠️  Skipping invalid file: {json_file.name}")

        except json.JSONDecodeError as e:
            print(f"❌ Error: {json_file.name} is not valid JSON: {e}")
        except Exception as e:
            print(f"❌ Error reading {json_file.name}: {e}")

    return submissions


def load_leaderboard(leaderboard_file: Path) -> List[Dict[str, Any]]:
    """Load existing leaderboard data"""
    if not leaderboard_file.exists():
        print("ℹ️  leaderboard.json does not exist, will create new file")
        return []

    try:
        with open(leaderboard_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"📊 Current leaderboard has {len(data)} record(s)")
        return data
    except json.JSONDecodeError:
        print("⚠️  Warning: leaderboard.json format error, will recreate")
        return []
    except Exception as e:
        print(f"❌ Error reading leaderboard.json: {e}")
        return []


def calculate_total_score(metrics: Dict[str, float]) -> float:
    """
    Calculate total score from metrics
    R (Retrieval Score) = (recall + f1 + ndcg) / 3
    G (Generation Score) = correctness * 0.75 + faithfulness * 0.25
    Total Score = R * 0.25 + G * 0.75
    """
    R = (metrics['recall'] + metrics['f1'] + metrics['ndcg']) / 3
    G = metrics['correctness'] * 0.75 + metrics['faithfulness'] * 0.25
    total_score = R * 0.25 + G * 0.75
    return total_score


def merge_data(existing: List[Dict[str, Any]],
               new_submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge data with deduplication and update logic

    Rules:
    - If system name is same, keep the one with higher average total score
    """
    # Use dictionary for storage, key is system name
    systems_dict = {}

    # Add existing data first
    for item in existing:
        system_name = item['system_name']
        systems_dict[system_name] = item

    # Process new submissions
    updates = 0
    additions = 0

    for item in new_submissions:
        system_name = item['system_name']

        if system_name in systems_dict:
            existing_item = systems_dict[system_name]

            # Calculate total scores for comparison
            new_total = calculate_total_score(item['average'])
            existing_total = calculate_total_score(existing_item['average'])

            # Update if new submission has higher average total score
            if new_total > existing_total:
                systems_dict[system_name] = item
                updates += 1
                print(f"🔄 Updated system: {system_name} (higher score: {new_total:.4f} vs {existing_total:.4f})")
            else:
                print(f"ℹ️  Keeping existing record: {system_name} (same or lower score)")
        else:
            systems_dict[system_name] = item
            additions += 1
            print(f"➕ Added new system: {system_name}")

    print(f"\n📈 Statistics: Added {additions}, Updated {updates}")

    # Convert back to list
    return list(systems_dict.values())


def save_leaderboard(data: List[Dict[str, Any]], leaderboard_file: Path):
    """Save leaderboard data as JSON and embed into index.html for file:// compatibility"""
    # Sort by calculated total score descending
    data.sort(key=lambda x: calculate_total_score(x['average']), reverse=True)

    try:
        json_str = json.dumps(data, ensure_ascii=False, indent=2)

        # Save leaderboard.json
        with open(leaderboard_file, 'w', encoding='utf-8') as f:
            f.write(json_str)

        # Embed data into index.html so the page works when opened via file:// protocol
        import re
        index_file = leaderboard_file.with_name('index.html')
        if index_file.exists():
            html = index_file.read_text(encoding='utf-8')
            # Replace the inline script between the marker comments
            pattern = r'(<!-- LEADERBOARD_DATA_START -->\n\s*<script>).*?(</script>\n\s*<!-- LEADERBOARD_DATA_END -->)'
            compact_json = json.dumps(data, ensure_ascii=False)
            replacement = rf'\1window.__LEADERBOARD_DATA__ = {compact_json};\2'
            new_html = re.sub(pattern, replacement, html, flags=re.DOTALL)
            index_file.write_text(new_html, encoding='utf-8')
            print(f"✅ Embedded data into index.html")

        print(f"\n✅ Successfully saved to {leaderboard_file}")
        print(f"📊 Leaderboard now has {len(data)} record(s)")
    except Exception as e:
        print(f"❌ Error saving leaderboard.json: {e}")
        raise


def main():
    print("=" * 60)
    print("🚀 Starting ChronoPlay RAG Leaderboard Aggregation")
    print("=" * 60)

    # Define paths
    base_dir = Path(__file__).parent
    submissions_dir = base_dir / 'submissions'
    leaderboard_file = base_dir / 'leaderboard.json'

    # Load data
    existing_data = load_leaderboard(leaderboard_file)
    new_submissions = load_submissions(submissions_dir)

    if not new_submissions:
        print("\n✅ No new submissions, no update needed")
        return

    # Merge data
    print("\n" + "=" * 60)
    print("🔄 Merging Data")
    print("=" * 60)
    merged_data = merge_data(existing_data, new_submissions)

    # Save results
    print("\n" + "=" * 60)
    print("💾 Saving Results")
    print("=" * 60)
    save_leaderboard(merged_data, leaderboard_file)

    print("\n" + "=" * 60)
    print("✅ Aggregation Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
