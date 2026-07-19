import Link from "next/link";

export default function Navbar() {
    return (
        <nav className="border-b border-slate-200 bg-white/80 text-slate-900 shadow-sm backdrop-blur dark:border-slate-800 dark:bg-slate-950/80 dark:text-white sticky top-0 z-40">
            <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-4 sm:px-6 lg:flex-row lg:items-center lg:justify-between lg:gap-6 lg:px-8">
                <ul className="flex flex-wrap items-center gap-x-6 gap-y-3 sm:gap-x-8">
                    <li>
                        <Link href="/" className="font-medium text-slate-600 transition-colors hover:text-slate-950 dark:text-slate-300 dark:hover:text-white">
                            Home
                        </Link>
                    </li>
                    <li>
                        <Link href="/explore" className="font-medium text-slate-600 transition-colors hover:text-slate-950 dark:text-slate-300 dark:hover:text-white">
                            Explore
                        </Link>
                    </li>
                    <li>
                        <Link href="/history" className="font-medium text-slate-600 transition-colors hover:text-slate-950 dark:text-slate-300 dark:hover:text-white">
                            History
                        </Link>
                    </li>
                    <li>
                        <Link href="/liked" className="font-medium text-slate-600 transition-colors hover:text-slate-950 dark:text-slate-300 dark:hover:text-white">
                            Liked
                        </Link>
                    </li>
                </ul>
            </div>
        </nav>
    );
};
