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

# 1. Create real_tenants table
con.run("""
CREATE TABLE IF NOT EXISTS public.real_tenants (
    tenant_id text PRIMARY KEY,
    user_id uuid REFERENCES auth.users(id) ON DELETE SET NULL,
    name text NOT NULL,
    api_key text,
    api_secret text,
    base_url text DEFAULT 'https://api.india.delta.exchange',
    ws_url text DEFAULT 'wss://socket.india.delta.exchange',
    initial_balance numeric DEFAULT 0.0,
    balance numeric DEFAULT 0.0,
    equity numeric DEFAULT 0.0,
    realized_pnl numeric DEFAULT 0.0,
    unrealized_pnl numeric DEFAULT 0.0,
    leverage integer DEFAULT 5,
    risk_pct numeric DEFAULT 1.0,
    timeframe text DEFAULT '1m',
    max_drawdown_pct numeric DEFAULT 4.0,
    symbols jsonb DEFAULT '["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "XAUTUSD"]'::jsonb,
    is_running boolean DEFAULT true,
    is_active boolean DEFAULT true,
    created_at timestamp with time zone DEFAULT timezone('utc'::text, now()),
    updated_at timestamp with time zone DEFAULT timezone('utc'::text, now())
);
""")

# 2. Create demo_tenants table
con.run("""
CREATE TABLE IF NOT EXISTS public.demo_tenants (
    tenant_id text PRIMARY KEY,
    user_id uuid REFERENCES auth.users(id) ON DELETE SET NULL,
    name text NOT NULL,
    api_key text,
    api_secret text,
    base_url text DEFAULT 'https://cdn-ind.testnet.deltaex.org',
    ws_url text DEFAULT 'wss://socket-ind.testnet.deltaex.org',
    initial_balance numeric DEFAULT 0.0,
    balance numeric DEFAULT 0.0,
    equity numeric DEFAULT 0.0,
    realized_pnl numeric DEFAULT 0.0,
    unrealized_pnl numeric DEFAULT 0.0,
    leverage integer DEFAULT 5,
    risk_pct numeric DEFAULT 1.0,
    timeframe text DEFAULT '1m',
    max_drawdown_pct numeric DEFAULT 4.0,
    symbols jsonb DEFAULT '["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "XAUTUSD"]'::jsonb,
    is_running boolean DEFAULT true,
    is_active boolean DEFAULT true,
    created_at timestamp with time zone DEFAULT timezone('utc'::text, now()),
    updated_at timestamp with time zone DEFAULT timezone('utc'::text, now())
);
""")

# 3. Add unique constraints on user_id for strict isolation
con.run("CREATE UNIQUE INDEX IF NOT EXISTS uq_real_tenants_user_id ON public.real_tenants (user_id) WHERE user_id IS NOT NULL")
con.run("CREATE UNIQUE INDEX IF NOT EXISTS uq_demo_tenants_user_id ON public.demo_tenants (user_id) WHERE user_id IS NOT NULL")

# 4. Populate real_tenants and demo_tenants from existing tenants table
con.run("""
INSERT INTO public.real_tenants (
    tenant_id, user_id, name, api_key, api_secret, base_url, ws_url,
    balance, equity, leverage, risk_pct, timeframe, max_drawdown_pct, symbols, is_running, is_active
)
SELECT 
    tenant_id, user_id, name, delta_api_key, delta_api_secret,
    COALESCE(delta_base_url, 'https://api.india.delta.exchange'),
    COALESCE(delta_ws_url, 'wss://socket.india.delta.exchange'),
    balance, equity, leverage, risk_pct, timeframe, max_drawdown_pct, symbols, is_running, is_active
FROM public.tenants
ON CONFLICT (tenant_id) DO UPDATE SET
    api_key = EXCLUDED.api_key,
    api_secret = EXCLUDED.api_secret,
    symbols = EXCLUDED.symbols;

INSERT INTO public.demo_tenants (
    tenant_id, user_id, name, api_key, api_secret, base_url, ws_url,
    balance, equity, leverage, risk_pct, timeframe, max_drawdown_pct, symbols, is_running, is_active
)
SELECT 
    tenant_id, user_id, name, demo_api_key, demo_api_secret,
    COALESCE(demo_base_url, 'https://cdn-ind.testnet.deltaex.org'),
    COALESCE(demo_ws_url, 'wss://socket-ind.testnet.deltaex.org'),
    balance, equity, leverage, risk_pct, timeframe, max_drawdown_pct, symbols, is_running, is_active
FROM public.tenants
ON CONFLICT (tenant_id) DO UPDATE SET
    api_key = EXCLUDED.api_key,
    api_secret = EXCLUDED.api_secret,
    symbols = EXCLUDED.symbols;
""")

# 5. Update user provisioning trigger to create rows in both separate tables
con.run("""
CREATE OR REPLACE FUNCTION public.handle_new_user_tenant()
 RETURNS trigger
 LANGUAGE plpgsql
 SECURITY DEFINER
AS $function$
DECLARE
  v_slug text;
  v_name text;
BEGIN
  IF EXISTS (SELECT 1 FROM public.real_tenants WHERE user_id = NEW.id) THEN
    RETURN NEW;
  END IF;

  v_name := COALESCE(NEW.raw_user_meta_data->>'full_name', split_part(NEW.email, '@', 1));
  v_slug := lower(regexp_replace(v_name, '[^a-zA-Z0-9]+', '-', 'g'));
  v_slug := trim(both '-' from v_slug);
  IF v_slug = '' OR v_slug = 'default' THEN
    v_slug := 'user-' || substr(NEW.id::text, 1, 8);
  END IF;

  IF EXISTS (SELECT 1 FROM public.real_tenants WHERE tenant_id = v_slug) OR EXISTS (SELECT 1 FROM public.demo_tenants WHERE tenant_id = v_slug) THEN
    v_slug := v_slug || '-' || substr(NEW.id::text, 1, 6);
  END IF;

  -- Insert into separate real_tenants table
  INSERT INTO public.real_tenants (
    tenant_id, user_id, name, base_url, ws_url, symbols
  ) VALUES (
    v_slug, NEW.id, v_name || '''s Real Trading Desk',
    'https://api.india.delta.exchange', 'wss://socket.india.delta.exchange',
    $json$["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "XAUTUSD"]$json$::jsonb
  ) ON CONFLICT (tenant_id) DO NOTHING;

  -- Insert into separate demo_tenants table
  INSERT INTO public.demo_tenants (
    tenant_id, user_id, name, base_url, ws_url, symbols
  ) VALUES (
    v_slug, NEW.id, v_name || '''s Demo Trading Desk',
    'https://cdn-ind.testnet.deltaex.org', 'wss://socket-ind.testnet.deltaex.org',
    $json$["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "XAUTUSD"]$json$::jsonb
  ) ON CONFLICT (tenant_id) DO NOTHING;

  RETURN NEW;
END;
$function$;
""")

rows_real = con.run("SELECT tenant_id, user_id, name, base_url FROM public.real_tenants")
rows_demo = con.run("SELECT tenant_id, user_id, name, base_url FROM public.demo_tenants")
print("Successfully created separated tables in Supabase!")
print(f"real_tenants count: {len(rows_real)}")
for r in rows_real:
    print(" [REAL]", r)
print(f"demo_tenants count: {len(rows_demo)}")
for r in rows_demo:
    print(" [DEMO]", r)
