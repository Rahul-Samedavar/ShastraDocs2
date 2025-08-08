import aiohttp
import asyncio
import tempfile
import os
import re
from urllib.parse import urlparse
from typing import List

class FileDownloader:

    async def download_file(self, url: str, timeout: int = 300, max_retries: int = 3) -> List[str]:
        """Download any file type from a URL to a temporary file with enhanced error handling."""
        print(f"📥 Downloading file from: {url[:60]}...")

        for attempt in range(max_retries):
            try:
                timeout_config = aiohttp.ClientTimeout(
                    total=timeout,
                    connect=30,
                    sock_read=120
                )

                async with aiohttp.ClientSession(timeout=timeout_config) as session:
                    print(f"   Attempt {attempt + 1}/{max_retries} (timeout: {timeout}s)")

                    async with session.get(url) as response:
                        if response.status != 200:
                            raise Exception(f"Failed to download file: HTTP {response.status}")

                        # Extract filename from header or URL
                        cd = response.headers.get('Content-Disposition', '')
                        filename_match = re.findall('filename="?([^"]+)"?', cd)
                        if filename_match:
                            filename = filename_match[0]
                        else:
                            from urllib.parse import unquote
                            path = urlparse(url).path
                            filename = os.path.basename(unquote(path))  # Decode URL encoding

                        if not filename:
                            filename = "downloaded_file"

                        ext = os.path.splitext(filename)[1]
                        if not ext:
                            ext = ".bin"  # Fallback if no extension is found

                        print(f"   📁 Detected filename: {filename}, extension: {ext}")

                        if ext not in ['.pdf', '.docx', '.pptx', '.png', '.xlsx', '.jpeg', '.jpg', '.txt', '.csv']:
                            # Return extension without dot for consistency
                            ext_without_dot = ext[1:] if ext.startswith('.') else ext
                            print(f"   ❌ File type not supported: {ext}")
                            return ['not supported', ext_without_dot]

                        # Get content length
                        content_length = response.headers.get('content-length')
                        if content_length:
                            total_size = int(content_length)
                            print(f"   File size: {total_size / (1024 * 1024):.1f} MB")

                        # Create temp file with same extension
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext, prefix="download_")

                        # Write to file
                        downloaded = 0
                        async for chunk in response.content.iter_chunked(16384):
                            temp_file.write(chunk)
                            downloaded += len(chunk)

                            if content_length and downloaded % (1024 * 1024) == 0:
                                progress = (downloaded / total_size) * 100
                                print(f"   Progress: {progress:.1f}% ({downloaded / (1024*1024):.1f} MB)")

                        temp_file.close()
                        print(f"✅ File downloaded successfully: {temp_file.name}")
                        # Return extension without the dot for consistency with modular_preprocessor
                        ext_without_dot = ext[1:] if ext.startswith('.') else ext
                        return temp_file.name, ext_without_dot

            except asyncio.TimeoutError:
                print(f"   ⏰ Timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 30
                    print(f"   ⏳ Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                continue

            except Exception as e:
                print(f"   ❌ Error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 15
                    print(f"   ⏳ Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                continue

        raise Exception(f"Failed to download file after {max_retries} attempts")

    def cleanup_temp_file(self, temp_path: str) -> None:
        """
        Clean up temporary file.
        
        Args:
            temp_path: Path to the temporary file to delete
        """
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                print(f"🗑️ Cleaned up temporary file: {temp_path}")
            except Exception as e:
                print(f"⚠️ Warning: Could not delete temporary file {temp_path}: {e}")