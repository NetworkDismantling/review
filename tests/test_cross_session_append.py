"""Test cross-session append functionality for Parquet writer."""
import multiprocessing as mp
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import tempfile
import shutil

from network_dismantling.common.storage.pandas.parquet import start_df_writer


def test_cross_session_append():
    """Test that cross-session append works by:
    1. Write initial batch (session 1)
    2. Close writer
    3. Write another batch in append mode (session 2)
    4. Verify both batches are present
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test_append.parquet"
        
        # Session 1: Write initial data
        print("\n=== Session 1: Initial write ===")
        queue1 = mp.Queue()
        
        df1 = pd.DataFrame({
            'network': ['net1'] * 3,
            'run': [1, 1, 1],
            'removals': [
                [{'removal_num': 0, 'id': 1, 'prediction': 0.9, 'lcc_size': 100, 'slcc_size': 10}],
                [{'removal_num': 1, 'id': 5, 'prediction': 0.8, 'lcc_size': 90, 'slcc_size': 15}],
                [{'removal_num': 2, 'id': 10, 'prediction': 0.7, 'lcc_size': 80, 'slcc_size': 20}],
            ]
        })
        
        writer1 = start_df_writer(
            output_file=output_file,
            output_df_columns=['network', 'run', 'removals'],
            df_queue=queue1,
            mode='overwrite',
        )
        
        queue1.put(df1)
        queue1.put(None)  # Sentinel
        writer1.join()
        
        # Verify session 1 wrote correctly
        table1 = pq.read_table(str(output_file))
        print(f"After session 1: {table1.num_rows} rows")
        assert table1.num_rows == 3, f"Expected 3 rows, got {table1.num_rows}"
        
        # Session 2: Append more data
        print("\n=== Session 2: Append mode ===")
        queue2 = mp.Queue()
        
        df2 = pd.DataFrame({
            'network': ['net1'] * 2,
            'run': [2, 2],
            'removals': [
                [{'removal_num': 0, 'id': 2, 'prediction': 0.95, 'lcc_size': 100, 'slcc_size': 8}],
                [{'removal_num': 1, 'id': 7, 'prediction': 0.85, 'lcc_size': 92, 'slcc_size': 12}],
            ]
        })
        
        writer2 = start_df_writer(
            output_file=output_file,
            output_df_columns=['network', 'run', 'removals'],
            df_queue=queue2,
            mode='append',  # Cross-session append!
        )
        
        queue2.put(df2)
        queue2.put(None)  # Sentinel
        writer2.join()
        
        # Verify both sessions' data is present
        final_table = pq.read_table(str(output_file))
        print(f"After session 2: {final_table.num_rows} rows")
        assert final_table.num_rows == 5, f"Expected 5 rows (3+2), got {final_table.num_rows}"
        
        # Verify schema is still efficient struct format
        removals_field = final_table.schema.field('removals')
        removals_type_str = str(removals_field.type)
        print(f"Removals schema: {removals_type_str}")
        
        # Check for efficient struct format (not string)
        assert 'struct' in removals_type_str.lower(), f"Expected struct format, got {removals_type_str}"
        assert 'string' not in removals_type_str.lower(), f"Got string format (inefficient): {removals_type_str}"
        
        # Verify data integrity
        df_final = final_table.to_pandas()
        assert len(df_final[df_final['run'] == 1]) == 3, "Session 1 data missing"
        assert len(df_final[df_final['run'] == 2]) == 2, "Session 2 data missing"
        
        print("\n✅ Cross-session append test PASSED")
        print(f"   - Session 1: 3 rows written")
        print(f"   - Session 2: 2 rows appended")
        print(f"   - Final: {final_table.num_rows} rows total")
        print(f"   - Schema: {removals_type_str}")


def test_multiple_appends():
    """Test multiple cross-session appends."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test_multi_append.parquet"
        
        total_rows = 0
        
        # Write 3 sessions
        for session in range(1, 4):
            print(f"\n=== Session {session} ===")
            queue = mp.Queue()
            
            df = pd.DataFrame({
                'session': [session] * 2,
                'value': [session * 10 + i for i in range(2)],
            })
            
            mode = 'overwrite' if session == 1 else 'append'
            writer = start_df_writer(
                output_file=output_file,
                output_df_columns=['session', 'value'],
                df_queue=queue,
                mode=mode,
            )
            
            queue.put(df)
            queue.put(None)
            writer.join()
            
            total_rows += len(df)
        
        # Verify all sessions present
        final_table = pq.read_table(str(output_file))
        print(f"\nFinal: {final_table.num_rows} rows")
        assert final_table.num_rows == 6, f"Expected 6 rows (2+2+2), got {final_table.num_rows}"
        
        df_final = final_table.to_pandas()
        for session in range(1, 4):
            session_data = df_final[df_final['session'] == session]
            assert len(session_data) == 2, f"Session {session} data missing or incomplete"
        
        print("✅ Multiple appends test PASSED")


if __name__ == "__main__":
    test_cross_session_append()
    test_multiple_appends()
    print("\n" + "="*60)
    print("All cross-session append tests PASSED! 🎉")
    print("="*60)
