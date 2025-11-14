import asyncio
import cognee

async def main():
    await cognee.add("Test document for Cognee with Gemini")
    await cognee.cognify()
    results = await cognee.search("What is this document about?")
    print(results)

asyncio.run(main())
