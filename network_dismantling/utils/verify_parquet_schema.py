#!/usr/bin/env python3
"""Verify that Parquet files use efficient struct format for removals column."""

import sys
from pathlib import Path
import pyarrow.parquet as pq


def verify_parquet_schema(file_path: Path):
    """Verify that removals column is stored as list<struct>, not list<string> or dict.
    
    Efficient format: list<struct<removal_num: uint32, id: uint32, prediction: float32, lcc_size: uint32, slcc_size: uint32>>
    Inefficient formats: list<string>, list<binary>, etc.
    """
    print(f"\n{'='*80}")
    print(f"Verificando schema di: {file_path.name}")
    print(f"{'='*80}\n")
    
    # Leggi solo lo schema, senza caricare dati
    schema = pq.read_schema(str(file_path))
    
    print("Schema completo del file:")
    print(schema)
    print()
    
    # Verifica colonna removals
    if 'removals' not in schema.names:
        print("⚠️  Colonna 'removals' non trovata nello schema")
        return False
    
    removals_idx = schema.names.index('removals')
    removals_type = schema.types[removals_idx]
    
    print(f"Tipo della colonna 'removals': {removals_type}")
    print(f"Tipo PyArrow: {type(removals_type)}")
    print()
    
    # Verifica che sia list<struct<...>>
    if str(removals_type).startswith('list<struct'):
        print("✅ CORRETTO: removals è salvato come list<struct>")
        print("   Lo schema è definito una volta, i valori sono colonnari.")
        print("   Questo è efficiente in spazio e performance.")
        
        # Mostra i campi della struct
        import pyarrow as pa
        if isinstance(removals_type, pa.ListType):
            value_type = removals_type.value_type
            if isinstance(value_type, pa.StructType):
                print(f"\n   Campi della struct:")
                for field in value_type:
                    print(f"     - {field.name}: {field.type}")
        return True
    
    elif 'string' in str(removals_type).lower() or 'binary' in str(removals_type).lower():
        print("❌ INEFFICIENTE: removals è salvato come stringa/binary")
        print("   I dati sono serializzati come testo, non strutturati.")
        print("   Le chiavi vengono ripetute per ogni elemento (spreco di spazio).")
        return False
    
    else:
        print(f"⚠️  FORMATO SCONOSCIUTO: {removals_type}")
        return False


def verify_file_size(file_path: Path):
    """Mostra dimensione del file e statistiche."""
    size_mb = file_path.stat().st_size / 1024 / 1024
    print(f"\n📊 Dimensione file: {size_mb:.2f} MB")
    
    # Leggi metadata per contare righe
    metadata = pq.read_metadata(str(file_path))
    num_rows = metadata.num_rows
    num_row_groups = metadata.num_row_groups
    
    print(f"   Righe totali: {num_rows:,}")
    print(f"   Row groups: {num_row_groups}")
    print(f"   Dimensione media per riga: {size_mb * 1024 * 1024 / num_rows:.1f} bytes")
    
    # Mostra compression per colonna
    print(f"\n   Compression per colonna:")
    for rg_idx in range(min(1, num_row_groups)):  # Solo primo row group
        rg = metadata.row_group(rg_idx)
        for col_idx in range(rg.num_columns):
            col = rg.column(col_idx)
            path = col.path_in_schema
            uncompressed = col.total_uncompressed_size
            compressed = col.total_compressed_size
            ratio = uncompressed / compressed if compressed > 0 else 0
            
            if 'removals' in path:
                print(f"     {path}: {compressed/1024:.1f} KB → {uncompressed/1024:.1f} KB (ratio {ratio:.2f}x)")


def compare_schemas(efficient_file: Path, inefficient_file: Path = None):
    """Confronta schema di due file per vedere differenze."""
    if not inefficient_file or not inefficient_file.exists():
        return
    
    print(f"\n{'='*80}")
    print("CONFRONTO TRA FILE EFFICIENTE E INEFFICIENTE")
    print(f"{'='*80}\n")
    
    schema1 = pq.read_schema(str(efficient_file))
    schema2 = pq.read_schema(str(inefficient_file))
    
    print(f"File 1 (efficiente): {efficient_file.name}")
    print(f"  removals type: {schema1.field('removals').type}\n")
    
    print(f"File 2 (inefficiente): {inefficient_file.name}")
    print(f"  removals type: {schema2.field('removals').type}\n")
    
    # Confronta dimensioni
    size1 = efficient_file.stat().st_size / 1024 / 1024
    size2 = inefficient_file.stat().st_size / 1024 / 1024
    saving = (1 - size1 / size2) * 100 if size2 > 0 else 0
    
    print(f"Dimensioni:")
    print(f"  File efficiente:   {size1:.2f} MB")
    print(f"  File inefficiente: {size2:.2f} MB")
    print(f"  Risparmio:         {saving:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python verify_parquet_schema.py <file.parquet> [file2.parquet]")
        print("\nEsempio:")
        print("  python verify_parquet_schema.py results.parquet")
        print("  python verify_parquet_schema.py efficient.parquet inefficient.parquet")
        sys.exit(1)
    
    file1 = Path(sys.argv[1])
    if not file1.exists():
        print(f"❌ File non trovato: {file1}")
        sys.exit(1)
    
    # Verifica primo file
    is_efficient = verify_parquet_schema(file1)
    verify_file_size(file1)
    
    # Confronta con secondo file se fornito
    if len(sys.argv) > 2:
        file2 = Path(sys.argv[2])
        if file2.exists():
            print()
            verify_parquet_schema(file2)
            verify_file_size(file2)
            compare_schemas(file1, file2)
    
    sys.exit(0 if is_efficient else 1)
