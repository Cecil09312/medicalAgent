"""
网络搜索工具
支持 DuckDuckGo 搜索和网页内容抓取
用于医疗信息检索
"""
import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger

try:
    import httpx
    from bs4 import BeautifulSoup
    WEB_SCRAPING_ENABLED = True
except ImportError:
    logger.warning("httpx or beautifulsoup4 not installed, web scraping disabled")
    WEB_SCRAPING_ENABLED = False


@dataclass
class SearchResult:
    """搜索结果数据结构"""
    title: str
    url: str
    snippet: str
    source: str = "web"


class WebSearchTool:
    """
    网络搜索工具
    支持 DuckDuckGo 搜索，可选内容抓取
    """

    def __init__(self, timeout: int = 30):
        """
        初始化搜索工具

        Args:
            timeout: 请求超时时间（秒）
        """
        self.timeout = timeout
        # 医疗领域可信域名白名单
        self.medical_domains = [
            "pubmed.ncbi.nlm.nih.gov",
            "mayoclinic.org",
            "webmd.com",
            "who.int",
            "cdc.gov",
            "nih.gov",
            "uptodate.com",
            "medscape.com",
            "healthline.com",
            "medicalnewstoday.com",
        ]
        logger.debug(f"WebSearchTool initialized with {len(self.medical_domains)} medical domains")

    async def search(
        self,
        query: str,
        max_results: int = 10,
        region: str = "cn-zh",
        safesearch: str = "on",
        timelimit: Optional[str] = None,
        retry_count: int = 2,
    ) -> List[SearchResult]:
        """
        执行网络搜索

        Args:
            query: 搜索查询
            max_results: 最大结果数
            region: 地区代码
            safesearch: 安全搜索级别
            timelimit: 时间限制（如 "d", "w", "m", "y"）
            retry_count: 重试次数

        Returns:
            搜索结果列表
        """
        # 尝试导入 DDGS（支持不同版本的包名）
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                logger.error("Neither 'ddgs' nor 'duckduckgo_search' package found")
                return []

        results = []

        # 尝试多个后端
        backends = ["bing", "duckduckgo", "auto"]

        for backend in backends:
            for attempt in range(retry_count + 1):
                try:
                    logger.debug(f"Searching with backend={backend}, attempt={attempt + 1}")

                    with DDGS() as ddgs:
                        search_params = {
                            "keywords": query,
                            "max_results": max_results,
                            "region": region,
                            "safesearch": safesearch,
                        }
                        if timelimit:
                            search_params["timelimit"] = timelimit

                        # 执行搜索
                        ddgs_results = list(ddgs.text(**search_params))

                        # 转换为 SearchResult
                        for r in ddgs_results:
                            results.append(SearchResult(
                                title=r.get("title", ""),
                                url=r.get("href", r.get("url", "")),
                                snippet=r.get("body", r.get("snippet", "")),
                                source=backend,
                            ))

                        logger.info(f"Found {len(results)} results from {backend}")
                        return results

                except Exception as e:
                    logger.warning(f"Search failed (backend={backend}, attempt={attempt + 1}): {e}")
                    if attempt < retry_count:
                        await asyncio.sleep(1)
                    continue

            logger.warning(f"Backend {backend} failed after {retry_count + 1} attempts")

        logger.error("All search backends failed")
        return results

    def filter_by_domain(
        self,
        results: List[SearchResult],
        allowed_domains: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        按域名过滤结果

        Args:
            results: 搜索结果列表
            allowed_domains: 允许的域名列表（默认使用医疗域名白名单）

        Returns:
            过滤后的结果列表
        """
        if allowed_domains is None:
            allowed_domains = self.medical_domains

        filtered = []
        for result in results:
            for domain in allowed_domains:
                if domain in result.url:
                    filtered.append(result)
                    break

        logger.debug(f"Filtered {len(results)} results to {len(filtered)} by domain")
        return filtered

    async def fetch_content(self, url: str, max_length: int = 2000) -> Optional[str]:
        """
        抓取网页内容

        Args:
            url: 网页URL
            max_length: 最大内容长度

        Returns:
            网页文本内容
        """
        if not WEB_SCRAPING_ENABLED:
            logger.warning("Web scraping not available (missing dependencies)")
            return None

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")

                # 移除 script 和 style 标签
                for script in soup(["script", "style"]):
                    script.decompose()

                # 提取文本内容
                text = soup.get_text(separator=" ", strip=True)

                # 截断
                if len(text) > max_length:
                    text = text[:max_length] + "..."

                logger.debug(f"Fetched {len(text)} chars from {url}")
                return text

        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    async def search_with_content(
        self,
        query: str,
        max_results: int = 5,
        fetch_full_content: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        搜索并可选抓取完整内容

        Args:
            query: 搜索查询
            max_results: 最大结果数
            fetch_full_content: 是否抓取完整内容

        Returns:
            包含搜索结果和可选内容的列表
        """
        results = await self.search(query, max_results=max_results)

        enriched_results = []
        for result in results:
            enriched = {
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet,
                "source": result.source,
            }

            if fetch_full_content:
                content = await self.fetch_content(result.url)
                enriched["content"] = content

            enriched_results.append(enriched)

        return enriched_results


def search_medical_web() -> WebSearchTool:
    """
    便捷函数：创建医疗搜索工具实例

    Returns:
        WebSearchTool 实例
    """
    return WebSearchTool()
