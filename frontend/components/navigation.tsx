import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import { Settings, Search } from 'lucide-react';

export function Navigation() {
  const pathname = usePathname();

  return (
    <div className="fixed top-4 left-4 z-50">
      <nav className="flex items-center gap-2">
        <Link
          href="/"
          className={cn(
            "px-4 py-2 rounded-full text-sm font-medium transition-all duration-200",
            "border border-transparent hover:border-primary/50",
            pathname === '/' 
              ? "bg-gradient-to-r from-purple-500 to-pink-500 text-white" 
              : "bg-card/95 backdrop-blur-sm text-foreground hover:bg-card"
          )}
        >
          <Search className="w-4 h-4 inline-block mr-2" />
          Search
        </Link>
        <Link
          href="/admin"
          className={cn(
            "px-4 py-2 rounded-full text-sm font-medium transition-all duration-200",
            "border border-transparent hover:border-primary/50",
            pathname === '/admin' 
              ? "bg-gradient-to-r from-purple-500 to-pink-500 text-white" 
              : "bg-card/95 backdrop-blur-sm text-foreground hover:bg-card"
          )}
        >
          <Settings className="w-4 h-4 inline-block mr-2" />
          Admin
        </Link>
      </nav>
    </div>
  );
}
