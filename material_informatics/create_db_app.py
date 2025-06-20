import streamlit as st
import PyPDF2
import sqlite3
import tempfile
import os
import re
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize session state for tracking uploaded files
if "last_uploaded_files" not in st.session_state:
    st.session_state.last_uploaded_files = []
if "db_created" not in st.session_state:
    st.session_state.db_created = False

def extract_metadata_and_content(pdf_file):
    """Extract title, authors, year, and content from a PDF file."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_file_path = tmp_file.name
        
        pdf_reader = PyPDF2.PdfReader(tmp_file_path)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        os.unlink(tmp_file_path)
        
        if not text.strip():
            logger.warning(f"No text extracted from {pdf_file.name}")
            return {
                "title": pdf_file.name,
                "authors": "Unknown",
                "year": str(datetime.now().year),
                "content": f"No text extracted from {pdf_file.name}."
            }
        
        # Attempt to extract metadata
        title = pdf_file.name  # Default to filename
        authors = "Unknown"
        year = str(datetime.now().year)  # Default to current year
        
        # Try PDF metadata
        try:
            metadata = pdf_reader.metadata
            if metadata:
                if "/Title" in metadata and metadata["/Title"]:
                    title = metadata["/Title"]
                if "/Author" in metadata and metadata["/Author"]:
                    authors = metadata["/Author"]
        except Exception as e:
            logger.warning(f"Error accessing PDF metadata for {pdf_file.name}: {e}")
        
        # Try extracting from text
        try:
            # Look for title (first line or pattern like "Title: ...")
            lines = text.splitlines()
            if lines:
                first_line = lines[0].strip()
                if len(first_line) > 10 and not first_line.lower().startswith(("abstract", "introduction")):
                    title = first_line[:100]  # Limit title length
            title_match = re.search(r"(?:Title|TITLE)\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip()[:100]
            
            # Look for authors (e.g., "Author(s): ...", or names before abstract)
            author_match = re.search(r"(?:Author|Authors|By)\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
            if author_match:
                authors = author_match.group(1).strip()
            elif "abstract" in text.lower():
                pre_abstract = text[:text.lower().index("abstract")].strip()
                name_pattern = r"[A-Z][a-z]+(?:\s[A-Z]\.)?(?:\s[A-Z][a-z]+)?(?:,\s[A-Z][a-z]+(?:\s[A-Z]\.)?(?:\s[A-Z][a-z]+)?)*"
                author_match = re.search(name_pattern, pre_abstract)
                if author_match:
                    authors = author_match.group(0).strip()
            
            # Look for year (four-digit number or "Year: ...")
            year_match = re.search(r"(?:Year|Published)\s*:\s*(\d{4})", text, re.IGNORECASE)
            if year_match:
                year = year_match.group(1)
            else:
                year_match = re.search(r"\b(20\d{2})\b", text)
                if year_match:
                    year = year_match.group(1)
        except Exception as e:
            logger.warning(f"Error extracting metadata from text for {pdf_file.name}: {e}")
        
        logger.info(f"Extracted metadata for {pdf_file.name}: Title={title}, Authors={authors}, Year={year}")
        return {
            "title": title,
            "authors": authors,
            "year": year,
            "content": text
        }
    except Exception as e:
        logger.error(f"Error processing {pdf_file.name}: {e}")
        return {
            "title": pdf_file.name,
            "authors": "Unknown",
            "year": str(datetime.now().year),
            "content": f"Error extracting text from {pdf_file.name}: {str(e)}"
        }

def create_database(papers, db_name="papers.db"):
    """Create a SQLite database and store the parsed papers."""
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        # Create table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                authors TEXT,
                year TEXT,
                content TEXT
            )
        """)
        
        # Clear existing data
        cursor.execute("DELETE FROM papers")
        
        # Insert parsed papers
        for paper in papers:
            cursor.execute("""
                INSERT INTO papers (title, authors, year, content)
                VALUES (?, ?, ?, ?)
            """, (paper["title"], paper["authors"], paper["year"], paper["content"]))
        
        conn.commit()
        conn.close()
        logger.info(f"Database '{db_name}' created with {len(papers)} papers")
        return db_name
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        raise

def main():
    st.set_page_config(page_title="PDF to SQLite Database Converter", layout="wide")
    st.title("PDF to SQLite Database Converter")
    st.markdown("""
    Upload one or more PDF files to extract text and create a SQLite database (.db) file compatible with LiₓSnᵧ phase NER analysis.
    The database will be updated whenever new PDFs are uploaded, overwriting the previous version.
    """)
    
    # Display log output
    log_container = st.empty()
    log_buffer = []
    
    def update_log(message):
        log_buffer.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")
        if len(log_buffer) > 20:
            log_buffer.pop(0)
        log_container.text_area("Processing Logs", "\n".join(log_buffer), height=200)
    
    # File uploader
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        # Check if uploaded files have changed
        current_filenames = sorted([f.name for f in uploaded_files])
        last_filenames = sorted(st.session_state.last_uploaded_files)
        if current_filenames != last_filenames or not st.session_state.db_created:
            st.session_state.last_uploaded_files = current_filenames
            st.session_state.db_created = False
            
            with st.spinner("Processing PDFs and creating database..."):
                papers = []
                for uploaded_file in uploaded_files:
                    update_log(f"Processing {uploaded_file.name}...")
                    paper_data = extract_metadata_and_content(uploaded_file)
                    papers.append(paper_data)
                    update_log(f"Extracted: Title={paper_data['title']}, Authors={paper_data['authors']}, Year={paper_data['year']}, Content length={len(paper_data['content'])}")
                
                if not papers:
                    update_log("No valid papers extracted from PDFs.")
                    st.error("No valid papers extracted from PDFs. Please check the files.")
                    return
                
                try:
                    db_name = "papers.db"
                    create_database(papers, db_name)
                    st.session_state.db_created = True
                    update_log(f"Database '{db_name}' created successfully with {len(papers)} papers.")
                    st.success(f"Database '{db_name}' created with {len(papers)} papers!")
                    
                    # Display extracted papers
                    st.subheader("Extracted Papers")
                    for i, paper in enumerate(papers, 1):
                        with st.expander(f"Paper {i}: {paper['title']}"):
                            st.write(f"**Title**: {paper['title']}")
                            st.write(f"**Authors**: {paper['authors']}")
                            st.write(f"**Year**: {paper['year']}")
                            st.write(f"**Content Preview**: {paper['content'][:200]}...")
                    
                    # Provide download link for the database
                    with open(db_name, "rb") as f:
                        db_bytes = f.read()
                    st.download_button(
                        label="Download Database File",
                        data=db_bytes,
                        file_name=db_name,
                        mime="application/x-sqlite3"
                    )
                    
                except Exception as e:
                    update_log(f"Error creating database: {str(e)}")
                    st.error(f"Error creating database: {str(e)}")
                
        else:
            st.info("No new PDFs uploaded. Using existing database.")
            if os.path.exists("papers.db"):
                with open("papers.db", "rb") as f:
                    db_bytes = f.read()
                st.download_button(
                    label="Download Existing Database File",
                    data=db_bytes,
                    file_name="papers.db",
                    mime="application/x-sqlite3"
                )
    
    else:
        st.info("Please upload PDF files to create a database.")
        st.session_state.db_created = False
        st.session_state.last_uploaded_files = []

if __name__ == "__main__":
    main()

