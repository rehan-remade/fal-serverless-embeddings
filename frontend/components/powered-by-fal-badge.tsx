import React from "react";
import Link from "next/link";
import { LogoIcon } from "@/components/logo";

export const PoweredByFalBadge: React.FC = () => {
  return (
    <div className="fixed top-4 right-4 z-50 hidden md:block">
      <Link
        href="https://fal.ai"
        target="_blank"
        className="border bg-card/95 backdrop-blur-sm p-2 flex flex-row rounded-xl gap-2 items-center hover:border-primary/50 transition-all duration-200"
      >
        <LogoIcon className="w-10 h-10" />
        <div className="text-xs">
          Powered by <br />
          <span className="font-bold text-xl">fal</span>
        </div>
      </Link>
    </div>
  );
};
