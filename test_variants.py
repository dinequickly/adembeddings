from ad_pipeline.src.pipeline import generate_variants
from ad_pipeline.src.brief import default_brief
import json

brief = default_brief()
results = generate_variants(brief, backend='qwen')

print(json.dumps(results, indent=2))
