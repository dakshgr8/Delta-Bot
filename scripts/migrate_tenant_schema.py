import os, urllib.parse, pg8000.native
from dotenv import load_dotenv

load_dotenv()
db_url = os.getenv("SUPABASE_DB_URL")
p = urllib.parse.urlparse(db_url)
con = pg8000.native.Connection(
    user=p.username,
    password=p.password,
    host=p.hostname,
    port=p.port or 5432,
    database=p.path.lstrip("/"),
    ssl_context=True,
)

# 1. Delete duplicate user-0141fe76
con.run("DELETE FROM public.tenants WHERE tenant_id = 'user-0141fe76'")

# 2. Update nitinyadavtss and dakshtiwari12345 with proper symbols and live Delta India URLs
con.run("""
ALTER TABLE public.tenants ADD COLUMN IF NOT EXISTS demo_api_key text;
ALTER TABLE public.tenants ADD COLUMN IF NOT EXISTS demo_api_secret text;
ALTER TABLE public.tenants ADD COLUMN IF NOT EXISTS demo_base_url text DEFAULT 'https://cdn-ind.testnet.deltaex.org';
ALTER TABLE public.tenants ADD COLUMN IF NOT EXISTS demo_ws_url text DEFAULT 'wss://socket-ind.testnet.deltaex.org';

UPDATE public.tenants
SET symbols = $json$["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "XAUTUSD"]$json$::jsonb,
    delta_base_url = 'https://api.india.delta.exchange',
    delta_ws_url = 'wss://socket.india.delta.exchange',
    demo_base_url = 'https://cdn-ind.testnet.deltaex.org',
    demo_ws_url = 'wss://socket-ind.testnet.deltaex.org'
WHERE tenant_id IN ('nitinyadavtss', 'dakshtiwari12345')
""")

# 3. Add unique index on user_id to ensure database-level uniqueness
con.run(
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_tenants_user_id ON public.tenants (user_id) WHERE user_id IS NOT NULL"
)

# 4. Update trigger function with correct jsonb symbols
trigger_sql = """
CREATE OR REPLACE FUNCTION public.handle_new_user_tenant()
 RETURNS trigger
 LANGUAGE plpgsql
 SECURITY DEFINER
AS $function$
DECLARE
  v_slug text;
  v_name text;
BEGIN
  -- 1. Prevent duplicate provisioning if user already has a registered tenant
  IF EXISTS (SELECT 1 FROM public.tenants WHERE user_id = NEW.id) THEN
    RETURN NEW;
  END IF;

  v_name := COALESCE(NEW.raw_user_meta_data->>'full_name', split_part(NEW.email, '@', 1));
  v_slug := lower(regexp_replace(v_name, '[^a-zA-Z0-9]+', '-', 'g'));
  v_slug := trim(both '-' from v_slug);
  IF v_slug = '' OR v_slug = 'default' THEN
    v_slug := 'user-' || substr(NEW.id::text, 1, 8);
  END IF;

  IF EXISTS (SELECT 1 FROM public.tenants WHERE tenant_id = v_slug) THEN
    v_slug := v_slug || '-' || substr(NEW.id::text, 1, 6);
  END IF;

  INSERT INTO public.tenants (
    tenant_id,
    user_id,
    name,
    mode,
    initial_balance,
    balance,
    equity,
    leverage,
    risk_pct,
    timeframe,
    max_drawdown_pct,
    delta_base_url,
    delta_ws_url,
    demo_base_url,
    demo_ws_url,
    symbols,
    is_running,
    is_active
  ) VALUES (
    v_slug,
    NEW.id,
    v_name || '''s Trading Desk',
    'live',
    0.00,
    0.00,
    0.00,
    5,
    1.00,
    '1m',
    4.00,
    'https://api.india.delta.exchange',
    'wss://socket.india.delta.exchange',
    'https://cdn-ind.testnet.deltaex.org',
    'wss://socket-ind.testnet.deltaex.org',
    $json$["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "XAUTUSD"]$json$::jsonb,
    true,
    true
  ) ON CONFLICT (tenant_id) DO NOTHING;

  RETURN NEW;
END;
$function$;
"""
con.run(trigger_sql)

rows = con.run("SELECT tenant_id, user_id, name, symbols FROM public.tenants")
print("Remaining tenants in Supabase:")
for r in rows:
    print(r)
print("Migration completed successfully!")
