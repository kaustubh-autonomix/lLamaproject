import asyncio
import os
import PyPDF2
import cognee
from dotenv import load_dotenv


async def main():
    load_dotenv()

    path = "/Users/apple/Downloads/Downloadreceipt.pdf"
    try:
        reader = PyPDF2.PdfReader(path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        if not text.strip():
            print("Error: No text could be extracted from the PDF.")
            print("Please check the PDF file path and content.")
            return

    except FileNotFoundError:
        print(f"Error: PDF file not found at path: {path}")
        print("Please make sure the file path is correct.")
        return
    except Exception as e:
        print(f"An error occurred while reading the PDF: {e}")
        return

    await cognee.add(text)

    await cognee.cognify()

    q = input("Query: ")
    r = await cognee.search(q)
    print("\nSearch Results:")
    print(r)


if __name__ == "__main__":
    asyncio.run(main())