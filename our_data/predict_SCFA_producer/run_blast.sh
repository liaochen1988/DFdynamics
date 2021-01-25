#!/bin/sh

#blastn -query asv-sequences.fasta -out asv_ncbi_microbial16S_100perc_hit.blast -task megablast -db ../../microbiome_database/16S_ribosomal_RNA/16S_ribosomal_RNA -num_threads 12 -outfmt 7 -max_target_seqs 100 -perc_identity 100

blastn -query asv-sequences.fasta -out asv_ncbi_microbial16S_top100_hits.blast -task megablast -db ../../microbiome_database/16S_ribosomal_RNA/16S_ribosomal_RNA -num_threads 12 -outfmt 7 -max_target_seqs 100
