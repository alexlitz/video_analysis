"""
XG File Downloader - Downloads XG files from various sources.
"""

import re
import requests
import zipfile
import io
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs


class XGDownloader:
    """Downloads XG files from various hosting services."""

    def __init__(self, output_dir: str = "data/xg_files"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

    def download(self, url: str, filename: Optional[str] = None) -> Optional[Path]:
        """
        Download XG file from URL.

        Supports:
        - Direct .xg file URLs
        - Google Drive links
        - Dropbox links
        - ZIP files containing XG files

        Args:
            url: URL to download from
            filename: Optional filename to save as

        Returns:
            Path to downloaded file or None if failed
        """
        parsed = urlparse(url)

        # Route to appropriate handler
        if "drive.google.com" in parsed.netloc:
            return self._download_google_drive(url, filename)
        elif "dropbox.com" in parsed.netloc:
            return self._download_dropbox(url, filename)
        else:
            return self._download_direct(url, filename)

    def _download_direct(self, url: str, filename: Optional[str] = None) -> Optional[Path]:
        """Download from direct URL."""
        try:
            response = self.session.get(url, timeout=60, stream=True)
            response.raise_for_status()

            # Determine filename
            if not filename:
                content_disp = response.headers.get("Content-Disposition", "")
                match = re.search(r'filename="?([^"]+)"?', content_disp)
                if match:
                    filename = match.group(1)
                else:
                    filename = url.split("/")[-1].split("?")[0]

            output_path = self.output_dir / filename

            # Check if it's a ZIP file
            content_type = response.headers.get("Content-Type", "")
            if ".zip" in filename.lower() or "zip" in content_type.lower():
                return self._extract_zip(response.content, filename)

            # Save direct file
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return output_path

        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return None

    def _download_google_drive(self, url: str, filename: Optional[str] = None) -> Optional[Path]:
        """Download from Google Drive."""
        # Extract file ID
        file_id = None

        if "/file/d/" in url:
            match = re.search(r"/file/d/([^/]+)", url)
            if match:
                file_id = match.group(1)
        elif "id=" in url:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            file_id = params.get("id", [None])[0]

        if not file_id:
            print(f"Could not extract file ID from: {url}")
            return None

        # Download using direct download URL
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

        try:
            response = self.session.get(download_url, timeout=60)

            # Handle virus scan warning for large files
            if "download_warning" in response.text:
                match = re.search(r'confirm=([^&]+)', response.text)
                if match:
                    confirm = match.group(1)
                    download_url = f"https://drive.google.com/uc?export=download&confirm={confirm}&id={file_id}"
                    response = self.session.get(download_url, timeout=60)

            if not filename:
                filename = f"gdrive_{file_id}.xg"

            output_path = self.output_dir / filename

            # Check for ZIP
            if response.content[:4] == b'PK\x03\x04':
                return self._extract_zip(response.content, filename)

            with open(output_path, "wb") as f:
                f.write(response.content)

            return output_path

        except Exception as e:
            print(f"Failed to download from Google Drive: {e}")
            return None

    def _download_dropbox(self, url: str, filename: Optional[str] = None) -> Optional[Path]:
        """Download from Dropbox."""
        # Convert to direct download link
        if "?dl=0" in url:
            url = url.replace("?dl=0", "?dl=1")
        elif "?dl=1" not in url:
            url = url + ("&" if "?" in url else "?") + "dl=1"

        return self._download_direct(url, filename)

    def _extract_zip(self, content: bytes, source_name: str) -> Optional[Path]:
        """Extract XG files from a ZIP archive."""
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                xg_files = [
                    name for name in zf.namelist()
                    if name.lower().endswith('.xg')
                ]

                if not xg_files:
                    print(f"No XG files found in archive: {source_name}")
                    return None

                # Extract all XG files
                extracted = []
                for xg_file in xg_files:
                    output_path = self.output_dir / Path(xg_file).name
                    with zf.open(xg_file) as src, open(output_path, "wb") as dst:
                        dst.write(src.read())
                    extracted.append(output_path)
                    print(f"Extracted: {output_path}")

                return extracted[0] if len(extracted) == 1 else extracted

        except zipfile.BadZipFile:
            print(f"Invalid ZIP file: {source_name}")
            return None

    def download_batch(self, urls: list[str]) -> dict[str, Optional[Path]]:
        """Download multiple files."""
        results = {}
        for url in urls:
            results[url] = self.download(url)
        return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        downloader = XGDownloader()
        result = downloader.download(sys.argv[1])
        if result:
            print(f"Downloaded to: {result}")
        else:
            print("Download failed")
    else:
        print("Usage: python xg_downloader.py <url>")
