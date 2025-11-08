"""
Dummy Data Generator for IAM ML Framework
Generates realistic sample IAM data for testing and demonstration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


class IAMDataGenerator:
    """
    Generate realistic IAM data including:
    - Decision history (approval/rejection decisions)
    - Access usage (how frequently access is used)
    - Peer group (organizational context)
    """
    
    def __init__(self, n_users=500, n_access_items=100, n_requests=2000):
        """
        Initialize data generator.
        
        Args:
            n_users: Number of unique users
            n_access_items: Number of unique access items
            n_requests: Number of access requests
        """
        self.n_users = n_users
        self.n_access_items = n_access_items
        self.n_requests = n_requests
        
        # Generate base entities
        self.users = self._generate_user_ids()
        self.access_items = self._generate_access_items()
        self.roles = ['Developer', 'Manager', 'Admin', 'Analyst', 'Engineer', 
                     'Director', 'Consultant', 'Specialist']
        self.departments = ['IT', 'Finance', 'HR', 'Sales', 'Operations', 
                           'Marketing', 'Legal', 'Product']
        self.locations = ['New York', 'San Francisco', 'London', 'Singapore', 
                         'Chicago', 'Austin', 'Seattle', 'Boston']
        self.decisions = ['approve', 'reject']
        self.access_types = ['Role', 'Entitlement', 'Permission', 'Application']
    
    def _generate_user_ids(self):
        """Generate user IDs."""
        return [f"USER_{i:04d}" for i in range(1, self.n_users + 1)]
    
    def _generate_access_items(self):
        """Generate access item names."""
        apps = ['SAP', 'Salesforce', 'Workday', 'ServiceNow', 'Jira', 'GitHub', 
                'AWS', 'Azure', 'Oracle', 'Tableau']
        access_types = ['Admin', 'Read', 'Write', 'Execute', 'Delete', 'Approve']
        
        items = []
        for app in apps:
            for access_type in access_types:
                items.append(f"{app}_{access_type}")
        
        # Ensure we have enough items
        while len(items) < self.n_access_items:
            items.append(f"Access_Item_{len(items)}")
        
        return items[:self.n_access_items]
    
    def generate_decision_history(self):
        """
        Generate decision history dataset.
        
        Returns:
            DataFrame with decision history
        """
        print("Generating decision history...")
        
        data = []
        base_date = datetime.now() - timedelta(days=365)
        
        for i in range(self.n_requests):
            user_id = random.choice(self.users)
            access_item = random.choice(self.access_items)
            requester_role = random.choice(self.roles)
            approver_id = random.choice(self.users)
            
            # Risk score (influences decision)
            risk_score = np.random.beta(2, 5)  # Skewed towards lower risk
            
            # Decision logic (higher risk = more likely to reject)
            if risk_score > 0.7:
                decision = np.random.choice(self.decisions, p=[0.3, 0.7])  # More likely reject
            elif risk_score > 0.5:
                decision = np.random.choice(self.decisions, p=[0.6, 0.4])
            else:
                decision = np.random.choice(self.decisions, p=[0.85, 0.15])  # More likely approve
            
            # Timestamp
            timestamp = base_date + timedelta(
                days=random.randint(0, 365),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            data.append({
                'request_id': f"REQ_{i+1:06d}",
                'user_id': user_id,
                'access_item': access_item,
                'decision': decision,
                'timestamp': timestamp,
                'requester_role': requester_role,
                'approver_id': approver_id,
                'risk_score': round(risk_score, 4)
            })
        
        df = pd.DataFrame(data)
        print(f"  [OK] Generated {len(df)} decision records")
        print(f"  - Approvals: {(df['decision'] == 'approve').sum()}")
        print(f"  - Rejections: {(df['decision'] == 'reject').sum()}")
        
        return df
    
    def generate_access_usage(self):
        """
        Generate access usage dataset.
        
        Returns:
            DataFrame with access usage data
        """
        print("\nGenerating access usage data...")
        
        data = []
        base_date = datetime.now()
        
        # Generate usage for random user-access combinations
        n_usage_records = int(self.n_users * 1.5)  # Some users have multiple access items
        
        for _ in range(n_usage_records):
            user_id = random.choice(self.users)
            access_item = random.choice(self.access_items)
            
            # Frequency follows a power law (some access used heavily, most used rarely)
            frequency = int(np.random.pareto(1.5) * 10)
            frequency = max(0, min(frequency, 1000))  # Cap at 1000
            
            # Last used date (more frequent = more recent)
            if frequency > 50:
                days_ago = random.randint(0, 30)
            elif frequency > 10:
                days_ago = random.randint(0, 90)
            else:
                days_ago = random.randint(0, 180)
            
            last_used = base_date - timedelta(days=days_ago)
            
            access_type = random.choice(self.access_types)
            
            data.append({
                'user_id': user_id,
                'access_item': access_item,
                'last_used_date': last_used,
                'frequency': frequency,
                'access_type': access_type
            })
        
        df = pd.DataFrame(data)
        # Remove duplicates (user_id, access_item should be unique)
        df = df.drop_duplicates(subset=['user_id', 'access_item'])
        
        print(f"  [OK] Generated {len(df)} access usage records")
        print(f"  - Mean frequency: {df['frequency'].mean():.1f}")
        print(f"  - Median frequency: {df['frequency'].median():.1f}")
        
        return df
    
    def generate_peer_group(self):
        """
        Generate peer group / organizational context dataset.
        
        Returns:
            DataFrame with peer group data
        """
        print("\nGenerating peer group data...")
        
        data = []
        
        for user_id in self.users:
            role = random.choice(self.roles)
            department = random.choice(self.departments)
            location = random.choice(self.locations)
            
            # Seniority level (1-5)
            if 'Director' in role or 'Manager' in role:
                seniority = random.randint(3, 5)
            else:
                seniority = random.randint(1, 4)
            
            # Peer group ID (users in same dept+location+seniority are peers)
            peer_group_id = f"{department}_{location}_L{seniority}"
            
            data.append({
                'user_id': user_id,
                'peer_group_id': peer_group_id,
                'role': role,
                'department': department,
                'seniority_level': seniority,
                'location': location
            })
        
        df = pd.DataFrame(data)
        print(f"  [OK] Generated {len(df)} peer group records")
        print(f"  - Unique peer groups: {df['peer_group_id'].nunique()}")
        print(f"  - Departments: {df['department'].nunique()}")
        print(f"  - Locations: {df['location'].nunique()}")
        
        return df
    
    def save_datasets(self, output_dir='sample_datasets'):
        """
        Generate and save all datasets.
        
        Args:
            output_dir: Directory to save CSV files
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "=" * 60)
        print("IAM DUMMY DATA GENERATOR")
        print("=" * 60)
        
        # Generate datasets
        decision_history = self.generate_decision_history()
        access_usage = self.generate_access_usage()
        peer_group = self.generate_peer_group()
        
        # Save to CSV
        print("\nSaving datasets...")
        
        decision_path = os.path.join(output_dir, 'decision_history.csv')
        decision_history.to_csv(decision_path, index=False)
        print(f"  [OK] Saved: {decision_path}")
        
        usage_path = os.path.join(output_dir, 'access_usage.csv')
        access_usage.to_csv(usage_path, index=False)
        print(f"  [OK] Saved: {usage_path}")
        
        peer_path = os.path.join(output_dir, 'peer_group.csv')
        peer_group.to_csv(peer_path, index=False)
        print(f"  [OK] Saved: {peer_path}")
        
        print("\n" + "=" * 60)
        print("DATA GENERATION COMPLETE")
        print("=" * 60)
        print(f"\nGenerated datasets:")
        print(f"  - Decision History: {len(decision_history)} records")
        print(f"  - Access Usage: {len(access_usage)} records")
        print(f"  - Peer Group: {len(peer_group)} records")
        print(f"\nFiles saved to: {output_dir}/")
        print("=" * 60)


def main():
    """Main function to generate dummy data."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate dummy IAM data for testing')
    parser.add_argument('--users', type=int, default=500, 
                       help='Number of users (default: 500)')
    parser.add_argument('--access-items', type=int, default=100, 
                       help='Number of access items (default: 100)')
    parser.add_argument('--requests', type=int, default=2000, 
                       help='Number of access requests (default: 2000)')
    parser.add_argument('--output-dir', type=str, default='sample_datasets',
                       help='Output directory for CSV files (default: sample_datasets)')
    
    args = parser.parse_args()
    
    # Generate data
    generator = IAMDataGenerator(
        n_users=args.users,
        n_access_items=args.access_items,
        n_requests=args.requests
    )
    
    generator.save_datasets(args.output_dir)


if __name__ == "__main__":
    main()

