#!/bin/bash
export PGPASSWORD='postgres'
DB_NAME="tc_cadet_dataset_db"

echo "[*] Creating database: $DB_NAME..."
sudo -u postgres psql <<EOF
-- Create database
CREATE DATABASE $DB_NAME;
EOF

echo "[*] Creating tables inside $DB_NAME..."
sudo -u postgres psql -d $DB_NAME <<EOF

-- Create event table
CREATE TABLE event_table
(
    src_node      varchar,
    src_index_id  varchar,
    operation     varchar,
    dst_node      varchar,
    dst_index_id  varchar,
    timestamp_rec bigint,
    _id           serial
);
ALTER TABLE event_table OWNER TO postgres;
CREATE UNIQUE INDEX event_table__id_uindex ON event_table (_id);
GRANT DELETE, INSERT, REFERENCES, SELECT, TRIGGER, TRUNCATE, UPDATE ON event_table TO postgres;

-- Create file_node_table
CREATE TABLE file_node_table
(
    node_uuid varchar NOT NULL,
    hash_id   varchar NOT NULL,
    path      varchar,
    CONSTRAINT file_node_table_pk PRIMARY KEY (node_uuid, hash_id)
);
ALTER TABLE file_node_table OWNER TO postgres;

-- Create netflow_node_table
CREATE TABLE netflow_node_table
(
    node_uuid varchar NOT NULL,
    hash_id   varchar NOT NULL,
    src_addr  varchar,
    src_port  varchar,
    dst_addr  varchar,
    dst_port  varchar,
    CONSTRAINT netflow_node_table_pk PRIMARY KEY (node_uuid, hash_id)
);
ALTER TABLE netflow_node_table OWNER TO postgres;

-- Create subject_node_table
CREATE TABLE subject_node_table
(
    node_uuid varchar,
    hash_id   varchar,
    exec      varchar
);
ALTER TABLE subject_node_table OWNER TO postgres;

-- Create node2id table
CREATE TABLE node2id
(
    hash_id   varchar NOT NULL PRIMARY KEY,
    node_type varchar,
    msg       varchar,
    index_id  bigint
);
ALTER TABLE node2id OWNER TO postgres;
CREATE UNIQUE INDEX node2id_hash_id_uindex ON node2id (hash_id);

EOF

echo "[✓] DONE! Database and tables created successfully."
