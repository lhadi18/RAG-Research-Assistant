# Edge Caching Field Notes (Demonstration Corpus)

This self-authored teaching note is invented for software demonstration. It is not a scientific publication.

## Cache behavior
An edge cache stores copies of frequently requested objects near users. A cache hit serves an object without consulting the origin; a cache miss requires an origin fetch. The hit ratio is the number of hits divided by total requests. A higher hit ratio can reduce origin traffic and user latency, although results depend on workload and network conditions.

## Time to live and invalidation
Time to live (TTL) limits how long an object may remain fresh. Short TTLs reduce the time stale content can be served but increase origin revalidation traffic. Purge-based invalidation removes selected objects before TTL expiry. Versioned URLs make content changes explicit and can avoid broad purge operations.

## Eviction and workload
Least Recently Used (LRU) evicts the object that has gone unused for the longest time. Least Frequently Used (LFU) evicts objects with low request frequency. Neither policy is universally best: LRU adapts to changing recency, while LFU can retain historically popular objects after demand shifts. Evaluation should use representative request traces and report hit ratio, latency, and origin load.

## Privacy and failure modes
Caching personalized or sensitive responses requires careful cache keys and authorization controls. Shared caches should not serve one user's private response to another. Stale-while-revalidate can serve an older response while refreshing it, but the acceptable staleness window is an application decision. A cache outage should not silently bypass access controls at the origin.
